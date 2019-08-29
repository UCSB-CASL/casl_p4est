#ifdef P4_TO_P8
#include "my_p8est_level_set.h"
#include <src/point3.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#else
#include "my_p4est_level_set.h"
#include <src/point2.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#endif

#include <src/casl_math.h>
#include "petsc_compatibility.h"
#include <petsclog.h>

#undef MAX
#undef MIN

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_level_set_reinit_1st_order;
extern PetscLogEvent log_my_p4est_level_set_reinit_2nd_order;
extern PetscLogEvent log_my_p4est_level_set_reinit_1st_time_2nd_space;
extern PetscLogEvent log_my_p4est_level_set_reinit_2nd_time_1st_space;
extern PetscLogEvent log_my_p4est_level_set_reinit_1_iter_1st_order;
extern PetscLogEvent log_my_p4est_level_set_reinit_1_iter_2nd_order;
extern PetscLogEvent log_my_p4est_level_set_extend_over_interface;
extern PetscLogEvent log_my_p4est_level_set_extend_over_interface_TVD;
extern PetscLogEvent log_my_p4est_level_set_extend_from_interface;
extern PetscLogEvent log_my_p4est_level_set_extend_from_interface_TVD;
extern PetscLogEvent log_my_p4est_level_set_compute_derivatives;
extern PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_1_iter;
extern PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_Vec;
extern PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_CF2;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

void my_p4est_level_set_t::reinitialize_One_Iteration_First_Order( std::vector<p4est_locidx_t>& map, double *p0, double *pn, double *pnp1, double limit )
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_1_iter_1st_order, 0, 0, 0, 0);

  quad_neighbor_nodes_of_node_t qnnn;
  for( size_t n_map=0; n_map<map.size(); ++n_map)
  {
    p4est_locidx_t n = map[n_map];

    if(fabs(p0[n]) <= EPS)
      pnp1[n] = 0;
    else if(fabs(p0[n]) <= limit)
    {

      ngbd->get_neighbors(n, qnnn);

      double p0_000, p0_m00, p0_p00, p0_0m0, p0_0p0 ;
      double p_000 , p_m00 , p_p00 , p_0m0 , p_0p0  ;
#ifdef P4_TO_P8
      double p0_00m, p0_00p;
      double p_00m,  p_00p ;
#endif
#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(p0, p0_000, p0_m00, p0_p00, p0_0m0, p0_0p0, p0_00m, p0_00p);
      qnnn.ngbd_with_quadratic_interpolation(pn, p_000 , p_m00 , p_p00 , p_0m0 , p_0p0 , p_00m , p_00p );
#else
      qnnn.ngbd_with_quadratic_interpolation(p0, p0_000, p0_m00, p0_p00, p0_0m0, p0_0p0);
      qnnn.ngbd_with_quadratic_interpolation(pn, p_000 , p_m00 , p_p00 , p_0m0 , p_0p0 );
#endif
      double s_p00 = qnnn.d_p00; double s_m00 = qnnn.d_m00;
      double s_0p0 = qnnn.d_0p0; double s_0m0 = qnnn.d_0m0;
#ifdef P4_TO_P8
      double s_00p = qnnn.d_00p; double s_00m = qnnn.d_00m;
#endif

      //---------------------------------------------------------------------
      // check if the node is near interface
      //---------------------------------------------------------------------
      if (    (p0_000*p0_m00<0) || (p0_000*p0_p00<0)
              || (p0_000*p0_0m0<0) || (p0_000*p0_0p0<0)
        #ifdef P4_TO_P8
              || (p0_000*p0_00m<0) || (p0_000*p0_00p<0)
        #endif
              )
      {
        if(p0_000*p0_m00<0) { s_m00 = -interface_Location(-s_m00, 0, p0_m00, p0_000); p_m00 = 0; }
        if(p0_000*p0_p00<0) { s_p00 =  interface_Location( s_p00, 0, p0_p00, p0_000); p_p00 = 0; }
        if(p0_000*p0_0m0<0) { s_0m0 = -interface_Location(-s_0m0, 0, p0_0m0, p0_000); p_0m0 = 0; }
        if(p0_000*p0_0p0<0) { s_0p0 =  interface_Location( s_0p0, 0, p0_0p0, p0_000); p_0p0 = 0; }
#ifdef P4_TO_P8
        if(p0_000*p0_00m<0) { s_00m = -interface_Location(-s_00m, 0, p0_00m, p0_000); p_00m = 0; }
        if(p0_000*p0_00p<0) { s_00p =  interface_Location( s_00p, 0, p0_00p, p0_000); p_00p = 0; }
#endif

        s_m00 = MAX(s_m00,EPS);
        s_p00 = MAX(s_p00,EPS);
        s_0m0 = MAX(s_0m0,EPS);
        s_0p0 = MAX(s_0p0,EPS);
#ifdef P4_TO_P8
        s_00p = MAX(s_00p,EPS);
        s_00m = MAX(s_00m,EPS);
#endif
      }

      //---------------------------------------------------------------------
      // Neumann boundary condition on the walls
      //---------------------------------------------------------------------
      /* first unclamp the node */
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n);
      p4est_indep_t unclamped_node = *node;
      p4est_node_unclamp((p4est_quadrant_t*)&unclamped_node);

      /* wall in the x direction */
      if(unclamped_node.x==0)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
        if(nb_tree_idx == tree_idx) { s_m00 = s_p00; p_m00 = p_p00; }
      }
      else if(unclamped_node.x==P4EST_ROOT_LEN)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
        if(nb_tree_idx == tree_idx) { s_p00 = s_m00; p_p00 = p_m00; }
      }

      /* wall in the y direction */
      if(unclamped_node.y==0)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0m0];
        if(nb_tree_idx == tree_idx) { s_0m0 = s_0p0; p_0m0 = p_0p0; }
      }
      else if(unclamped_node.y==P4EST_ROOT_LEN)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0p0];
        if(nb_tree_idx == tree_idx) { s_0p0 = s_0m0; p_0p0 = p_0m0; }
      }
#ifdef P4_TO_P8
      /* wall in the z direction */
      if(unclamped_node.z==0)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00m];
        if(nb_tree_idx == tree_idx) { s_00m = s_00p; p_00m = p_00p; }
      }
      else if(unclamped_node.z==P4EST_ROOT_LEN)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00p];
        if(nb_tree_idx == tree_idx) { s_00p = s_00m; p_00p = p_00m; }
      }
#endif
      //---------------------------------------------------------------------
      // First Order One-Sided Differecing
      //---------------------------------------------------------------------
      double px_p00 = (p_p00-p_000)/s_p00; double px_m00 = (p_000-p_m00)/s_m00;
      double py_0p0 = (p_0p0-p_000)/s_0p0; double py_0m0 = (p_000-p_0m0)/s_0m0;
#ifdef P4_TO_P8
      double pz_00p = (p_00p-p_000)/s_00p; double pz_00m = (p_000-p_00m)/s_00m;
#endif

      double sgn = (p0_000>0) ? 1 : -1;

      //---------------------------------------------------------------------
      // Upwind Scheme
      //---------------------------------------------------------------------
      double dt = MIN(s_m00,s_p00);
      dt = MIN(dt,s_0m0);
      dt = MIN(dt,s_0p0);
#ifdef P4_TO_P8
      dt = MIN(dt,s_00m);
      dt = MIN(dt,s_00p);
#endif
      dt = dt/2.;

      if(sgn>0) {
        if(px_p00>0) px_p00 = 0;
        if(px_m00<0) px_m00 = 0;
        if(py_0p0>0) py_0p0 = 0;
        if(py_0m0<0) py_0m0 = 0;
#ifdef P4_TO_P8
        if(pz_00p>0) pz_00p = 0;
        if(pz_00m<0) pz_00m = 0;
#endif
      } else {
        if(px_p00<0) px_p00 = 0;
        if(px_m00>0) px_m00 = 0;
        if(py_0p0<0) py_0p0 = 0;
        if(py_0m0>0) py_0m0 = 0;
#ifdef P4_TO_P8
        if(pz_00p<0) pz_00p = 0;
        if(pz_00m>0) pz_00m = 0;
#endif
      }

#ifdef P4_TO_P8
      pnp1[n] = p_000 - dt*sgn*(sqrt( MAX(px_p00*px_p00 , px_m00*px_m00) +
                                      MAX(py_0p0*py_0p0 , py_0m0*py_0m0) +
                                      MAX(pz_00p*pz_00p , pz_00m*pz_00m) ) - 1.);
#else
      pnp1[n] = p_000 - dt*sgn*(sqrt( MAX(px_p00*px_p00 , px_m00*px_m00) +
                                      MAX(py_0p0*py_0p0 , py_0m0*py_0m0) ) - 1.);
#endif
      if(p0_000*pnp1[n]<0) pnp1[n] *= -1;

      ierr = PetscLogFlops(17); CHKERRXX(ierr);
    }
    /* else : far away from the interface and not in the band ... nothing to do */
    else
      pnp1[n] = p0[n];
  }
  ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_1_iter_1st_order, 0, 0, 0, 0);
}

void my_p4est_level_set_t::reinitialize_One_Iteration_Second_Order( std::vector<p4est_locidx_t>& map,
                                                                    #ifdef P4_TO_P8
                                                                    const double *dxx0, const double *dyy0, const double *dzz0,
                                                                    const double *dxx,  const double *dyy,  const double *dzz,
                                                                    #else
                                                                    const double *dxx0, const double *dyy0,
                                                                    const double *dxx,  const double *dyy,
                                                                    #endif
                                                                    double *p0, double *pn, double *pnp1, double limit )
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_1_iter_2nd_order, 0, 0, 0, 0);

  quad_neighbor_nodes_of_node_t qnnn;
  for( size_t n_map=0; n_map<map.size(); ++n_map)
  {
    p4est_locidx_t n = map[n_map];

    if(fabs(p0[n]) < EPS) {
      pnp1[n] = 0;
    } else if(fabs(p0[n]) <= limit) {
      ngbd->get_neighbors(n, qnnn);

      double p0_000, p0_m00, p0_p00, p0_0m0, p0_0p0 ;
      double p_000 , p_m00 , p_p00 , p_0m0 , p_0p0  ;
#ifdef P4_TO_P8
      double p0_00m, p0_00p;
      double p_00m,  p_00p ;
#endif
#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(p0, p0_000, p0_m00, p0_p00, p0_0m0, p0_0p0, p0_00m, p0_00p);
      qnnn.ngbd_with_quadratic_interpolation(pn, p_000 , p_m00 , p_p00 , p_0m0 , p_0p0 , p_00m , p_00p );
#else
      qnnn.ngbd_with_quadratic_interpolation(p0, p0_000, p0_m00, p0_p00, p0_0m0, p0_0p0);
      qnnn.ngbd_with_quadratic_interpolation(pn, p_000 , p_m00 , p_p00 , p_0m0 , p_0p0 );
#endif

      double s_p00 = qnnn.d_p00; double s_m00 = qnnn.d_m00;
      double s_0p0 = qnnn.d_0p0; double s_0m0 = qnnn.d_0m0;
#ifdef P4_TO_P8
      double s_00p = qnnn.d_00p; double s_00m = qnnn.d_00m;
#endif

      //---------------------------------------------------------------------
      // Second Order derivatives
      //---------------------------------------------------------------------
      double pxx_000 = dxx[n];
      double pyy_000 = dyy[n];
#ifdef P4_TO_P8
      double pzz_000 = dzz[n];
#endif
      double pxx_m00 = qnnn.f_m00_linear(dxx);
      double pxx_p00 = qnnn.f_p00_linear(dxx);
      double pyy_0m0 = qnnn.f_0m0_linear(dyy);
      double pyy_0p0 = qnnn.f_0p0_linear(dyy);
#ifdef P4_TO_P8
      double pzz_00m = qnnn.f_00m_linear(dzz);
      double pzz_00p = qnnn.f_00p_linear(dzz);
#endif

      //---------------------------------------------------------------------
      // check if the node is near interface
      //---------------------------------------------------------------------
      if (  (p0_000*p0_m00<0) || (p0_000*p0_p00<0)
         || (p0_000*p0_0m0<0) || (p0_000*p0_0p0<0)
        #ifdef P4_TO_P8
         || (p0_000*p0_00m<0) || (p0_000*p0_00p<0)
        #endif
         )
      {
        double p0xx_000 = dxx0[n];
        double p0yy_000 = dyy0[n];
#ifdef P4_TO_P8
        double p0zz_000 = dzz0[n];
#endif
        double p0xx_m00 = qnnn.f_m00_linear(dxx0);
        double p0xx_p00 = qnnn.f_p00_linear(dxx0);
        double p0yy_0m0 = qnnn.f_0m0_linear(dyy0);
        double p0yy_0p0 = qnnn.f_0p0_linear(dyy0);
#ifdef P4_TO_P8
        double p0zz_00m = qnnn.f_00m_linear(dzz0);
        double p0zz_00p = qnnn.f_00p_linear(dzz0);
#endif

        if(p0_000*p0_m00<0) { s_m00 =-interface_Location_With_Second_Order_Derivative(-s_m00,   0,p0_m00,p0_000,p0xx_m00,p0xx_000); p_m00=0; }
        if(p0_000*p0_p00<0) { s_p00 = interface_Location_With_Second_Order_Derivative(    0,s_p00,p0_000,p0_p00,p0xx_000,p0xx_p00); p_p00=0; }
        if(p0_000*p0_0m0<0) { s_0m0 =-interface_Location_With_Second_Order_Derivative(-s_0m0,   0,p0_0m0,p0_000,p0yy_0m0,p0yy_000); p_0m0=0; }
        if(p0_000*p0_0p0<0) { s_0p0 = interface_Location_With_Second_Order_Derivative(    0,s_0p0,p0_000,p0_0p0,p0yy_000,p0yy_0p0); p_0p0=0; }
#ifdef P4_TO_P8
        if(p0_000*p0_00m<0) { s_00m =-interface_Location_With_Second_Order_Derivative(-s_00m,   0,p0_00m,p0_000,p0zz_00m,p0zz_000); p_00m=0; }
        if(p0_000*p0_00p<0) { s_00p = interface_Location_With_Second_Order_Derivative(    0,s_00p,p0_000,p0_00p,p0zz_000,p0zz_00p); p_00p=0; }
#endif

        s_m00 = MAX(s_m00,EPS);
        s_p00 = MAX(s_p00,EPS);
        s_0m0 = MAX(s_0m0,EPS);
        s_0p0 = MAX(s_0p0,EPS);
#ifdef P4_TO_P8
        s_00m = MAX(s_00m,EPS);
        s_00p = MAX(s_00p,EPS);
#endif
      }

      //---------------------------------------------------------------------
      // Neumann boundary condition on the walls
      //---------------------------------------------------------------------
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n);

      if (is_node_xmWall(p4est, node)) { s_m00 = s_p00; p_m00=p_p00; pxx_000 = pxx_m00 = pxx_p00 = 0; }
      if (is_node_xpWall(p4est, node)) { s_p00 = s_m00; p_p00=p_m00; pxx_000 = pxx_m00 = pxx_p00 = 0; }
      if (is_node_ymWall(p4est, node)) { s_0m0 = s_0p0; p_0m0=p_0p0; pyy_000 = pyy_0m0 = pyy_0p0 = 0; }
      if (is_node_ypWall(p4est, node)) { s_0p0 = s_0m0; p_0p0=p_0m0; pyy_000 = pyy_0m0 = pyy_0p0 = 0; }
#ifdef P4_TO_P8
      if (is_node_zmWall(p4est, node)) { s_00m = s_00p; p_00m=p_00p; pzz_000 = pzz_00m = pzz_00p = 0; }
      if (is_node_zpWall(p4est, node)) { s_00p = s_00m; p_00p=p_00m; pzz_000 = pzz_00m = pzz_00p = 0; }
#endif

      //---------------------------------------------------------------------
      // First Order One-Sided Differecing
      //---------------------------------------------------------------------
      double px_p00 = (p_p00-p_000)/s_p00; double px_m00 = (p_000-p_m00)/s_m00;
      double py_0p0 = (p_0p0-p_000)/s_0p0; double py_0m0 = (p_000-p_0m0)/s_0m0;
#ifdef P4_TO_P8
      double pz_00p = (p_00p-p_000)/s_00p; double pz_00m = (p_000-p_00m)/s_00m;
#endif

      //---------------------------------------------------------------------
      // Second Order One-Sided Differencing
      //---------------------------------------------------------------------
      pxx_m00 = MINMOD(pxx_m00,pxx_000);   px_m00 += 0.5*s_m00*(pxx_m00);
      pxx_p00 = MINMOD(pxx_p00,pxx_000);   px_p00 -= 0.5*s_p00*(pxx_p00);
      pyy_0m0 = MINMOD(pyy_0m0,pyy_000);   py_0m0 += 0.5*s_0m0*(pyy_0m0);
      pyy_0p0 = MINMOD(pyy_0p0,pyy_000);   py_0p0 -= 0.5*s_0p0*(pyy_0p0);
#ifdef P4_TO_P8
      pzz_00m = MINMOD(pzz_00m,pzz_000);   pz_00m += 0.5*s_00m*(pzz_00m);
      pzz_00p = MINMOD(pzz_00p,pzz_000);   pz_00p -= 0.5*s_00p*(pzz_00p);
#endif

      double sgn = (p0_000>0) ? 1 : -1;

      //---------------------------------------------------------------------
      // Upwind Scheme
      //---------------------------------------------------------------------
      double dt = MIN(s_m00,s_p00);
      dt = MIN(dt,s_0m0);
      dt = MIN(dt,s_0p0);
#ifdef P4_TO_P8
      dt = MIN(dt,s_00m);
      dt = MIN(dt,s_00p);
      dt /= 3.0;
#else
      dt /= 2.0;
#endif

      if(sgn>0) {
        if(px_p00>0) px_p00 = 0;
        if(px_m00<0) px_m00 = 0;
        if(py_0p0>0) py_0p0 = 0;
        if(py_0m0<0) py_0m0 = 0;
#ifdef P4_TO_P8
        if(pz_00p>0) pz_00p = 0;
        if(pz_00m<0) pz_00m = 0;
#endif
      } else {
        if(px_p00<0) px_p00 = 0;
        if(px_m00>0) px_m00 = 0;
        if(py_0p0<0) py_0p0 = 0;
        if(py_0m0>0) py_0m0 = 0;
#ifdef P4_TO_P8
        if(pz_00p<0) pz_00p = 0;
        if(pz_00m>0) pz_00m = 0;
#endif
      }

#ifdef P4_TO_P8
      pnp1[n] = p_000 - dt*sgn*(sqrt( MAX(px_p00*px_p00 , px_m00*px_m00) +
                                      MAX(py_0p0*py_0p0 , py_0m0*py_0m0) +
                                      MAX(pz_00p*pz_00p , pz_00m*pz_00m) ) - 1.);
#else
      pnp1[n] = p_000 - dt*sgn*(sqrt( MAX(px_p00*px_p00 , px_m00*px_m00) +
                                      MAX(py_0p0*py_0p0 , py_0m0*py_0m0) ) - 1.);
#endif
      if(p0_000*pnp1[n]<0) pnp1[n] *= -1;

      ierr = PetscLogFlops(30); CHKERRXX(ierr);
    }
    /* else : far away from the interface and not in the band ... nothing to do */
    else
      pnp1[n] = p0[n];
  }
  ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_1_iter_2nd_order, 0, 0, 0, 0);
}

void my_p4est_level_set_t::advect_in_normal_direction_one_iteration(std::vector<p4est_locidx_t> &map, const double* vn, double dt,
                                                                    const double *dxx,  const double *dyy,
                                                                    #ifdef P4_TO_P8
                                                                    const double *dzz,
                                                                    #endif
                                                                    const double *pn, double *pnp1)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_advect_in_normal_direction_1_iter, 0, 0, 0, 0);


  bool px = is_periodic(p4est, 0);
  bool py = is_periodic(p4est, 1);
#ifdef P4_TO_P8
  bool pz = is_periodic(p4est, 2);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  for( size_t n_map=0; n_map<map.size(); ++n_map)
  {
    p4est_locidx_t n    = map[n_map];

    ngbd->get_neighbors(n, qnnn);

    double p_000 , p_m00 , p_p00 , p_0m0 , p_0p0;
#ifdef P4_TO_P8
    double p_00m, p_00p;
#endif

#ifdef P4_TO_P8
    qnnn.ngbd_with_quadratic_interpolation(pn, p_000 , p_m00 , p_p00 , p_0m0 , p_0p0, p_00m, p_00p);
#else
    qnnn.ngbd_with_quadratic_interpolation(pn, p_000 , p_m00 , p_p00 , p_0m0 , p_0p0);
#endif
    double s_p00 = qnnn.d_p00; double s_m00 = qnnn.d_m00;
    double s_0p0 = qnnn.d_0p0; double s_0m0 = qnnn.d_0m0;
#ifdef P4_TO_P8
    double s_00p = qnnn.d_00p; double s_00m = qnnn.d_00m;
#endif

    //---------------------------------------------------------------------
    // Second Order derivatives
    //---------------------------------------------------------------------
    double pxx_000 = dxx[n];
    double pyy_000 = dyy[n];
#ifdef P4_TO_P8
    double pzz_000 = dzz[n];
#endif
    double pxx_m00 = qnnn.f_m00_linear(dxx);
    double pxx_p00 = qnnn.f_p00_linear(dxx);
    double pyy_0m0 = qnnn.f_0m0_linear(dyy);
    double pyy_0p0 = qnnn.f_0p0_linear(dyy);
#ifdef P4_TO_P8
    double pzz_00m = qnnn.f_00m_linear(dzz);
    double pzz_00p = qnnn.f_00p_linear(dzz);
#endif

    //---------------------------------------------------------------------
    // Neumann boundary condition on the walls
    //---------------------------------------------------------------------
    /* first unclamp the node */
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n);
    p4est_indep_t unclamped_node = *node;
    p4est_node_unclamp((p4est_quadrant_t*)&unclamped_node);

    /* wall in the x direction */
    if(unclamped_node.x==0)
    {
      p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
      p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_m00];
      if(!px && nb_tree_idx == tree_idx) { s_m00 = s_p00; p_m00=p_p00; pxx_000 = pxx_m00 = pxx_p00 = 0; }
    }
    else if(unclamped_node.x==P4EST_ROOT_LEN)
    {
      p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
      p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_p00];
      if(!px && nb_tree_idx == tree_idx) { s_p00 = s_m00; p_p00=p_m00; pxx_000 = pxx_m00 = pxx_p00 = 0; }
    }

    /* wall in the y direction */
    if(unclamped_node.y==0)
    {
      p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
      p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0m0];
      if(!py && nb_tree_idx == tree_idx) { s_0m0 = s_0p0; p_0m0=p_0p0; pyy_000 = pyy_0m0 = pyy_0p0 = 0; }
    }
    else if(unclamped_node.y==P4EST_ROOT_LEN)
    {
      p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
      p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_0p0];
      if(!py && nb_tree_idx == tree_idx) { s_0p0 = s_0m0; p_0p0=p_0m0; pyy_000 = pyy_0m0 = pyy_0p0 = 0; }
    }

#ifdef P4_TO_P8
    /* wall in the z direction */
    if(unclamped_node.z==0)
    {
      p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
      p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00m];
      if(!pz && nb_tree_idx == tree_idx) { s_00m = s_00p; p_00m=p_00p; pzz_000 = pzz_00m = pzz_00p = 0; }
    }
    else if(unclamped_node.z==P4EST_ROOT_LEN)
    {
      p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
      p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[P4EST_FACES*tree_idx + dir::f_00p];
      if(!pz && nb_tree_idx == tree_idx) { s_00p = s_00m; p_00p=p_00m; pzz_000 = pzz_00m = pzz_00p = 0; }
    }
#endif

    //---------------------------------------------------------------------
    // First Order One-Sided Differecing
    //---------------------------------------------------------------------
    double px_p00 = (p_p00-p_000)/s_p00; double px_m00 = (p_000-p_m00)/s_m00;
    double py_0p0 = (p_0p0-p_000)/s_0p0; double py_0m0 = (p_000-p_0m0)/s_0m0;
#ifdef P4_TO_P8
    double pz_00p = (p_00p-p_000)/s_00p; double pz_00m = (p_000-p_00m)/s_00m;
#endif

    //---------------------------------------------------------------------
    // Second Order One-Sided Differencing
    //---------------------------------------------------------------------
    pxx_m00 = MINMOD(pxx_m00,pxx_000);   px_m00 += 0.5*s_m00*(pxx_m00);
    pxx_p00 = MINMOD(pxx_p00,pxx_000);   px_p00 -= 0.5*s_p00*(pxx_p00);
    pyy_0m0 = MINMOD(pyy_0m0,pyy_000);   py_0m0 += 0.5*s_0m0*(pyy_0m0);
    pyy_0p0 = MINMOD(pyy_0p0,pyy_000);   py_0p0 -= 0.5*s_0p0*(pyy_0p0);
#ifdef P4_TO_P8
    pzz_00m = MINMOD(pzz_00m,pzz_000);   pz_00m += 0.5*s_00m*(pzz_00m);
    pzz_00p = MINMOD(pzz_00p,pzz_000);   pz_00p -= 0.5*s_00p*(pzz_00p);
#endif

    if(vn[n]>0) {
      if(px_p00>0) px_p00 = 0;
      if(px_m00<0) px_m00 = 0;
      if(py_0p0>0) py_0p0 = 0;
      if(py_0m0<0) py_0m0 = 0;
#ifdef P4_TO_P8
      if(pz_00p>0) pz_00p = 0;
      if(pz_00m<0) pz_00m = 0;
#endif
    } else {
      if(px_p00<0) px_p00 = 0;
      if(px_m00>0) px_m00 = 0;
      if(py_0p0<0) py_0p0 = 0;
      if(py_0m0>0) py_0m0 = 0;
#ifdef P4_TO_P8
      if(pz_00p<0) pz_00p = 0;
      if(pz_00m>0) pz_00m = 0;
#endif
    }

#ifdef P4_TO_P8
    pnp1[n] = p_000 - dt*vn[n]*(sqrt(px_p00*px_p00 + px_m00*px_m00 +
                                     py_0p0*py_0p0 + py_0m0*py_0m0 +
                                     pz_00p*pz_00p + pz_00m*pz_00m));
#else
    pnp1[n] = p_000 - dt*vn[n]*(sqrt(px_p00*px_p00 + px_m00*px_m00 +
                                     py_0p0*py_0p0 + py_0m0*py_0m0));
#endif


    ierr = PetscLogFlops(30); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_level_set_advect_in_normal_direction_1_iter, 0, 0, 0, 0);
}


void my_p4est_level_set_t::reinitialize_1st_order( Vec phi_petsc, int number_of_iteration, double limit )
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_1st_order, phi_petsc, 0, 0, 0); CHKERRXX(ierr);

  double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));

  Vec p1_petsc;
  double *p1, *phi;
  ierr = VecDuplicate(phi_petsc, &p1_petsc); CHKERRXX(ierr);
  ierr = VecGetArray(p1_petsc, &p1); CHKERRXX(ierr);
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    p0[n] = phi[n];

  IPMLogRegionBegin("reinit_1st_1st");
  for(int i=0; i<number_of_iteration; i++)
  {

    /* 1) processes the layer nodes */
    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    reinitialize_One_Iteration_First_Order( ngbd->layer_nodes, p0, phi, p1, limit);

    /* 2) initiate the communication for the ghost layer */
    ierr = VecGhostUpdateBegin(p1_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* 3) process the local nodes */
    reinitialize_One_Iteration_First_Order( ngbd->local_nodes, p0, phi, p1, limit);

    /* 4) finish receiving the ghost layer */
    ierr = VecGhostUpdateEnd(p1_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* 5) Copy data into phi */
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      phi[n] = p1[n];
  }
  IPMLogRegionEnd("reinit_1st_1st");

  /* restore arrays and destroy uneeded petsc objects */
  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);
  ierr = VecRestoreArray(p1_petsc, &p1); CHKERRXX(ierr);
  ierr = VecDestroy(p1_petsc); CHKERRXX(ierr);

  free(p0);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_1st_order, phi_petsc, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_level_set_t::reinitialize_2nd_order( Vec phi_petsc, int number_of_iteration, double limit )
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_2nd_order, phi_petsc, 0, 0, 0); CHKERRXX(ierr);
  Vec p1_petsc, p2_petsc;
  double *p1, *p2, *phi;
  ierr = VecCreateGhostNodes(p4est, nodes, &p1_petsc); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &p2_petsc); CHKERRXX(ierr);
  ierr = VecGetArray(p1_petsc,  &p1);  CHKERRXX(ierr);
  ierr = VecGetArray(p2_petsc,  &p2);  CHKERRXX(ierr);
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

  double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    p0[n] = phi[n];

  Vec dxx0_petsc, dyy0_petsc;
  double *dxx0, *dyy0;
  ierr = VecCreateGhostNodes(p4est, nodes, &dxx0_petsc); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &dyy0_petsc); CHKERRXX(ierr);

  Vec dxx_petsc, dyy_petsc;
  double *dxx, *dyy;
  ierr = VecCreateGhostNodes(p4est, nodes, &dxx_petsc); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &dyy_petsc); CHKERRXX(ierr);

#ifdef P4_TO_P8
  Vec dzz_petsc, dzz0_petsc;
  double *dzz, *dzz0;
  ierr = VecCreateGhostNodes(p4est, nodes, &dzz_petsc ); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &dzz0_petsc); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
  compute_derivatives(phi_petsc, dxx0_petsc, dyy0_petsc, dzz0_petsc);
#else
  compute_derivatives(phi_petsc, dxx0_petsc, dyy0_petsc);
#endif

  ierr = VecGetArray(dxx0_petsc, &dxx0); CHKERRXX(ierr);
  ierr = VecGetArray(dyy0_petsc, &dyy0); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecGetArray(dzz0_petsc, &dzz0); CHKERRXX(ierr);
#endif

  for(int i=0; i<number_of_iteration; i++)
  {

    /***** Step 1 of RK2: phi -> p1 *****/
    /* compute derivatives */
#ifdef P4_TO_P8
    compute_derivatives(phi_petsc, dxx_petsc, dyy_petsc, dzz_petsc);
#else
    compute_derivatives(phi_petsc, dxx_petsc, dyy_petsc);
#endif

    ierr = VecGetArray(dxx_petsc, &dxx); CHKERRXX(ierr);
    ierr = VecGetArray(dyy_petsc, &dyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(dzz_petsc, &dzz); CHKERRXX(ierr);
#endif

    IPMLogRegionBegin("reinit_2nd_2nd");
    /* 1) Preocess layer nodes */
    reinitialize_One_Iteration_Second_Order( ngbd->layer_nodes,
                                         #ifdef P4_TO_P8
                                             dxx0, dyy0, dzz0,
                                             dxx,  dyy,  dzz,
                                         #else
                                             dxx0, dyy0,
                                             dxx,  dyy,
                                         #endif
                                             p0, phi, p1, limit);

    /* 2) Begin update process for p1 */
    ierr = VecGhostUpdateBegin(p1_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* 3) Preocess local nodes */
    reinitialize_One_Iteration_Second_Order( ngbd->local_nodes,
                                         #ifdef P4_TO_P8
                                             dxx0, dyy0, dzz0,
                                             dxx,  dyy,  dzz,
                                         #else
                                             dxx0, dyy0,
                                             dxx,  dyy,
                                         #endif
                                             p0, phi, p1, limit);

    /* 4) End update process for p1 */
    ierr = VecGhostUpdateEnd(p1_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    IPMLogRegionEnd("reinit_2nd_2nd");

    ierr = VecRestoreArray(dxx_petsc, &dxx); CHKERRXX(ierr);
    ierr = VecRestoreArray(dyy_petsc, &dyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(dzz_petsc, &dzz); CHKERRXX(ierr);
#endif

    /* recompute derivatives */
#ifdef P4_TO_P8
    compute_derivatives(p1_petsc, dxx_petsc, dyy_petsc, dzz_petsc);
#else
    compute_derivatives(p1_petsc, dxx_petsc, dyy_petsc);
#endif

    ierr = VecGetArray(dxx_petsc, &dxx); CHKERRXX(ierr);
    ierr = VecGetArray(dyy_petsc, &dyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(dzz_petsc, &dzz); CHKERRXX(ierr);
#endif

    /***** Step 2 of RK2: p1 -> p2 *****/

    IPMLogRegionBegin("reinit_2nd_2nd");
    /* 1) Preocess layer nodes */
    reinitialize_One_Iteration_Second_Order( ngbd->layer_nodes,
                                         #ifdef P4_TO_P8
                                             dxx0, dyy0, dzz0,
                                             dxx,  dyy,  dzz,
                                         #else
                                             dxx0, dyy0,
                                             dxx,  dyy,
                                         #endif
                                             p0, p1, p2, limit);

    /* 2) Begin update process for p2 */
    ierr = VecGhostUpdateBegin(p2_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* 3) Preocess local nodes */
    reinitialize_One_Iteration_Second_Order( ngbd->local_nodes,
                                         #ifdef P4_TO_P8
                                             dxx0, dyy0, dzz0,
                                             dxx,  dyy,  dzz,
                                         #else
                                             dxx0, dyy0,
                                             dxx,  dyy,
                                         #endif
                                             p0, p1, p2, limit);

    /* 4) End update process for p2 */
    ierr = VecGhostUpdateEnd(p2_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    IPMLogRegionEnd("reinit_2nd_2nd");

    ierr = VecRestoreArray(dxx_petsc, &dxx); CHKERRXX(ierr);
    ierr = VecRestoreArray(dyy_petsc, &dyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(dzz_petsc, &dzz); CHKERRXX(ierr);
#endif

    /* update phi */
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      phi[n] = .5 * (phi[n] + p2[n]);
  }

  /* restore arrays and destroy uneeded petsc objects */
  ierr = VecRestoreArray(dxx0_petsc, &dxx0); CHKERRXX(ierr);
  ierr = VecRestoreArray(dyy0_petsc, &dyy0); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(dzz0_petsc, &dzz0); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArray(p1_petsc,   &p1);   CHKERRXX(ierr);
  ierr = VecRestoreArray(p2_petsc,   &p2);   CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_petsc,  &phi);  CHKERRXX(ierr);

  ierr = VecDestroy(dxx0_petsc); CHKERRXX(ierr);
  ierr = VecDestroy(dyy0_petsc); CHKERRXX(ierr);
  ierr = VecDestroy(dxx_petsc);  CHKERRXX(ierr);
  ierr = VecDestroy(dyy_petsc);  CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(dzz0_petsc); CHKERRXX(ierr);
  ierr = VecDestroy(dzz_petsc ); CHKERRXX(ierr);
#endif
  ierr = VecDestroy(p1_petsc);   CHKERRXX(ierr);
  ierr = VecDestroy(p2_petsc);   CHKERRXX(ierr);

  free(p0);
  ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_2nd_order, phi_petsc, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_level_set_t::perturb_level_set_function( Vec phi_petsc, double epsilon )
{
  PetscErrorCode ierr;
  double *phi_ptr;

  ierr = VecGetArray(phi_petsc, &phi_ptr); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    if(fabs(phi_ptr[n]) < epsilon)
    {
      if(phi_ptr[n] > 0) phi_ptr[n] =  epsilon;
      else               phi_ptr[n] = -epsilon;
    }
  }

  ierr = VecRestoreArray(phi_petsc, &phi_ptr); CHKERRXX(ierr);
}


void my_p4est_level_set_t::reinitialize_2nd_order_time_1st_order_space( Vec phi_petsc, int number_of_iteration, double limit )
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_2nd_time_1st_space, phi_petsc, 0, 0, 0); CHKERRXX(ierr);

  Vec p1_petsc, p2_petsc;
  double *p1, *p2, *phi;
  ierr = VecCreateGhostNodes(p4est, nodes, &p1_petsc); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &p2_petsc); CHKERRXX(ierr);
  ierr = VecGetArray(p1_petsc,  &p1);  CHKERRXX(ierr);
  ierr = VecGetArray(p2_petsc,  &p2);  CHKERRXX(ierr);
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

  double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    p0[n] = phi[n];

  IPMLogRegionBegin("reinit_2nd_1st");
  for(int i=0; i<number_of_iteration; i++)
  {
    /***** Step 1 of RK2: phi -> p1 *****/

    /* 1) Preocess layer nodes */
    reinitialize_One_Iteration_First_Order( ngbd->layer_nodes, p0, phi, p1, limit);

    /* 2) Begin update process for p1 */
    ierr = VecGhostUpdateBegin(p1_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* 3) Preocess local nodes */
    reinitialize_One_Iteration_First_Order( ngbd->local_nodes, p0, phi, p1, limit);

    /* 4) End update process for p1 */
    ierr = VecGhostUpdateEnd(p1_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /***** Step 2 of RK2: p1 -> p2 *****/

    /* 1) Preocess layer nodes */
    reinitialize_One_Iteration_First_Order( ngbd->layer_nodes, p0, p1, p2, limit);

    /* 2) Begin update process for p2 */
    ierr = VecGhostUpdateBegin(p2_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* 3) Preocess local nodes */
    reinitialize_One_Iteration_First_Order( ngbd->local_nodes, p0, p1, p2, limit);

    /* 4) End update process for p2 */
    ierr = VecGhostUpdateEnd(p2_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* update phi */
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      phi[n] = .5 * (phi[n] + p2[n]);
  }
  IPMLogRegionEnd("reinit_2nd_1st");

  /* restore arrays and destroy uneeded petsc objects */

  ierr = VecRestoreArray(p1_petsc,  &p1);  CHKERRXX(ierr);
  ierr = VecRestoreArray(p2_petsc,  &p2);  CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);
  ierr = VecDestroy(p1_petsc); CHKERRXX(ierr);
  ierr = VecDestroy(p2_petsc); CHKERRXX(ierr);

  free(p0);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_2nd_time_1st_space, phi_petsc, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_level_set_t::reinitialize_1st_order_time_2nd_order_space( Vec phi, int number_of_iteration, double limit )
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_1st_time_2nd_space, phi, 0, 0, 0); CHKERRXX(ierr);

  Vec p1;
  double *p1_p, *phi_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &p1); CHKERRXX(ierr);

  Vec p1_loc;
  ierr = VecGhostGetLocalForm(p1, &p1_loc); CHKERRXX(ierr);
  ierr = VecGetArray(p1_loc,  &p1_p);  CHKERRXX(ierr);

  Vec phi_loc;
  ierr = VecGhostGetLocalForm(phi, &phi_loc); CHKERRXX(ierr);
  ierr = VecGetArray(phi_loc, &phi_p); CHKERRXX(ierr);

  double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
  memcpy(p0, phi_p, nodes->indep_nodes.elem_count*sizeof(double));

  Vec dxx0, dyy0;
  double *dxx0_p, *dyy0_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &dxx0); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &dyy0); CHKERRXX(ierr);

  Vec dxx, dyy;
  double *dxx_p, *dyy_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &dxx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &dyy); CHKERRXX(ierr);

#ifdef P4_TO_P8
  Vec dzz, dzz0;
  double *dzz_p, *dzz0_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &dzz0); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &dzz ); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
  compute_derivatives(phi, dxx0, dyy0, dzz0);
#else
  compute_derivatives(phi, dxx0, dyy0);
#endif

  ierr = VecGetArray(dxx0, &dxx0_p); CHKERRXX(ierr);
  ierr = VecGetArray(dyy0, &dyy0_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecGetArray(dzz0, &dzz0_p); CHKERRXX(ierr);
#endif

  for(int i=0; i<number_of_iteration; i++)
  {
    /* compute derivatives */
#ifdef P4_TO_P8
    compute_derivatives(phi, dxx, dyy, dzz);
#else
    compute_derivatives(phi, dxx, dyy);
#endif

    ierr = VecGetArray(dxx, &dxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(dyy, &dyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(dzz, &dzz_p); CHKERRXX(ierr);
#endif

    IPMLogRegionBegin("reinit_1st_2nd");
    /* 1) Preocess layer nodes */
    reinitialize_One_Iteration_Second_Order( ngbd->layer_nodes,
                                         #ifdef P4_TO_P8
                                             dxx0_p, dyy0_p, dzz0_p,
                                             dxx_p,  dyy_p,  dzz_p,
                                         #else
                                             dxx0_p, dyy0_p,
                                             dxx_p,  dyy_p,
                                         #endif
                                             p0, phi_p, p1_p, limit);

    /* 2) Begin update process for p1 */
    ierr = VecGhostUpdateBegin(p1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* 3) Preocess local nodes */
    reinitialize_One_Iteration_Second_Order( ngbd->local_nodes,
                                         #ifdef P4_TO_P8
                                             dxx0_p, dyy0_p, dzz0_p,
                                             dxx_p,  dyy_p,  dzz_p,
                                         #else
                                             dxx0_p, dyy0_p,
                                             dxx_p,  dyy_p,
                                         #endif
                                             p0, phi_p, p1_p, limit);

    /* 4) End update process for p1 */
    ierr = VecGhostUpdateEnd(p1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    IPMLogRegionEnd("reinit_1st_2nd");

    ierr = VecRestoreArray(dxx, &dxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(dyy, &dyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(dzz, &dzz_p); CHKERRXX(ierr);
#endif

    /* update phi */
    memcpy(phi_p, p1_p, nodes->indep_nodes.elem_count*sizeof(double));
  }

  /* restore arrays and destroy uneeded petsc objects */
  ierr = VecRestoreArray(dxx0, &dxx0_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(dyy0, &dyy0_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(dzz0, &dzz0_p); CHKERRXX(ierr);
#endif

  ierr = VecGhostRestoreLocalForm(phi, &phi_loc); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_loc, &phi_p); CHKERRXX(ierr);

  ierr = VecGhostRestoreLocalForm(p1, &p1_loc); CHKERRXX(ierr);
  ierr = VecRestoreArray(p1_loc, &p1_p); CHKERRXX(ierr);

  ierr = VecDestroy(dxx0); CHKERRXX(ierr);
  ierr = VecDestroy(dyy0); CHKERRXX(ierr);
  ierr = VecDestroy(dxx);  CHKERRXX(ierr);
  ierr = VecDestroy(dyy);  CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(dzz0); CHKERRXX(ierr);
  ierr = VecDestroy(dzz ); CHKERRXX(ierr);
#endif
  ierr = VecDestroy(p1);   CHKERRXX(ierr);

  free(p0);
  ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_1st_time_2nd_space, phi, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_level_set_t::compute_derivatives( Vec phi_petsc, Vec dxx_petsc, Vec dyy_petsc, Vec dzz_petsc) const
#else
void my_p4est_level_set_t::compute_derivatives( Vec phi_petsc, Vec dxx_petsc, Vec dyy_petsc) const
#endif
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_compute_derivatives, 0, 0, 0, 0); CHKERRXX(ierr);

#ifdef P4_TO_P8
  ngbd->second_derivatives_central(phi_petsc, dxx_petsc, dyy_petsc, dzz_petsc);
#else
  ngbd->second_derivatives_central(phi_petsc, dxx_petsc, dyy_petsc);
#endif

  ierr = PetscLogEventEnd(log_my_p4est_level_set_compute_derivatives, 0, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
double my_p4est_level_set_t::advect_in_normal_direction(const CF_3& vn, Vec phi, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
double my_p4est_level_set_t::advect_in_normal_direction(const CF_2& vn, Vec phi, Vec phi_xx, Vec phi_yy)
#endif
{
  /* TODO: do not allocate memory for vn */
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_advect_in_normal_direction_CF2, 0, 0, 0, 0);

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
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz_); CHKERRXX(ierr);
#endif
#ifdef P4_TO_P8
    compute_derivatives(phi, phi_xx_, phi_yy_, phi_zz_);
#else
    compute_derivatives(phi, phi_xx_, phi_yy_);
#endif
    local_derivatives = true;
  }

  double *phi_p, *phi_xx_p, *phi_yy_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif

  /* compute dt based on CFL condition */
  double dt_local = DBL_MAX;
  double dt;
  std::vector<double> vn_vec(nodes->indep_nodes.elem_count);
  double *vn_p = &vn_vec[0];
  quad_neighbor_nodes_of_node_t qnnn;
  for (p4est_locidx_t n = 0; n<nodes->num_owned_indeps; ++n){
    ngbd->get_neighbors(n, qnnn);

    double xyzn[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyzn);

    double s_p00 = fabs(qnnn.d_p00); double s_m00 = fabs(qnnn.d_m00);
    double s_0p0 = fabs(qnnn.d_0p0); double s_0m0 = fabs(qnnn.d_0m0);
#ifdef P4_TO_P8
    double s_00p = fabs(qnnn.d_00p); double s_00m = fabs(qnnn.d_00m);
#endif
#ifdef P4_TO_P8
    double s_min = MIN(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0), MIN(s_00p, s_00m));
#else
    double s_min = MIN(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0));
#endif

    /* choose CFL = 0.8 ... just for fun! */
#ifdef P4_TO_P8
    vn_vec[n] = vn(xyzn[0], xyzn[1], xyzn[2]);
#else
    vn_vec[n] = vn(xyzn[0], xyzn[1]);
#endif
    dt_local = MIN(dt_local, 0.8*fabs(s_min/vn_vec[n]));
  }
  MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  Vec p1, p2;
  double *p1_p, *p2_p;
  ierr = VecDuplicate(phi_xx_, &p1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_yy_, &p2); CHKERRXX(ierr);

  ierr = VecGetArray(p1, &p1_p); CHKERRXX(ierr);
  ierr = VecGetArray(p2, &p2_p); CHKERRXX(ierr);


  memcpy(p1_p, phi_p, sizeof(double) * nodes->indep_nodes.elem_count);
  // layer nodes
  advect_in_normal_direction_one_iteration(ngbd->layer_nodes, vn_p, dt,
                                         #ifdef P4_TO_P8
                                           phi_xx_p, phi_yy_p, phi_zz_p,
                                         #else
                                           phi_xx_p, phi_yy_p,
                                         #endif
                                           p1_p, phi_p);
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  // local nodes
  advect_in_normal_direction_one_iteration(ngbd->local_nodes, vn_p, dt,
                                         #ifdef P4_TO_P8
                                           phi_xx_p, phi_yy_p, phi_zz_p,
                                         #else
                                           phi_xx_p, phi_yy_p,
                                         #endif
                                           p1_p, phi_p);
  ierr = VecGhostUpdateEnd  (phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* now phi(Lnp1, Bnp1, Gnp1) */
#ifdef P4_TO_P8
  compute_derivatives(phi, phi_xx_, phi_yy_, phi_zz_);
#else
  compute_derivatives(phi, phi_xx_, phi_yy_);
#endif

  // layer nodes
  advect_in_normal_direction_one_iteration(ngbd->layer_nodes, vn_p, dt,
                                         #ifdef P4_TO_P8
                                           phi_xx_p, phi_yy_p, phi_zz_p,
                                         #else
                                           phi_xx_p, phi_yy_p,
                                         #endif
                                           phi_p, p2_p);
  ierr = VecGhostUpdateBegin(p2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // local nodes
  advect_in_normal_direction_one_iteration(ngbd->local_nodes, vn_p, dt,
                                         #ifdef P4_TO_P8
                                           phi_xx_p, phi_yy_p, phi_zz_p,
                                         #else
                                           phi_xx_p, phi_yy_p,
                                         #endif
                                           phi_p, p2_p);
  ierr = VecGhostUpdateEnd  (p2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] = 0.5 * (p1_p[n] + p2_p[n]);

  /* restore arrays */
  ierr = VecRestoreArray(phi,     &phi_p);    CHKERRXX(ierr);
  ierr = VecRestoreArray(p1,      &p1_p);     CHKERRXX(ierr);
  ierr = VecRestoreArray(p2,      &p2_p);     CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif

  ierr = PetscLogFlops(nodes->num_owned_indeps * P4EST_DIM); CHKERRXX(ierr);

  if (local_derivatives){
    ierr = VecDestroy(phi_xx_); CHKERRXX(ierr);
    ierr = VecDestroy(phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(phi_zz_); CHKERRXX(ierr);
#endif
  }

  ierr = VecDestroy(p1); CHKERRXX(ierr);
  ierr = VecDestroy(p2); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_advect_in_normal_direction_CF2, 0, 0, 0, 0);

  return dt;
}

#ifdef P4_TO_P8
double my_p4est_level_set_t::advect_in_normal_direction(const Vec vn, Vec phi, double dt_max, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
double my_p4est_level_set_t::advect_in_normal_direction(const Vec vn, Vec phi, double dt_max, Vec phi_xx, Vec phi_yy)
#endif
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_advect_in_normal_direction_Vec, 0, 0, 0, 0);

  double *vn_p;
  ierr = VecGetArray(vn, &vn_p); CHKERRXX(ierr);

  Vec phi_xx_ = phi_xx, phi_yy_ = phi_yy;
#ifdef P4_TO_P8
  Vec phi_zz_ = phi_zz;
#endif
  bool local_derivatives = false;
#ifdef P4_TO_P8
  if (phi_xx_ == NULL && phi_yy_ == NULL && phi_zz_ == NULL)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz_); CHKERRXX(ierr);
    compute_derivatives(phi, phi_xx_, phi_yy_, phi_zz_);
    local_derivatives = true;
  }
#else
  if (phi_xx_ == NULL && phi_yy_ == NULL)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy_); CHKERRXX(ierr);
    compute_derivatives(phi, phi_xx_, phi_yy_);
    local_derivatives = true;
  }
#endif

  double *phi_p, *phi_xx_p, *phi_yy_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif

  /* compute dt based on CFL condition */
  double dt_local = dt_max;
  double dt;
  quad_neighbor_nodes_of_node_t qnnn;
  for (p4est_locidx_t n = 0; n<nodes->num_owned_indeps; ++n){

    ngbd->get_neighbors(n, qnnn);

    double s_p00 = fabs(qnnn.d_p00); double s_m00 = fabs(qnnn.d_m00);
    double s_0p0 = fabs(qnnn.d_0p0); double s_0m0 = fabs(qnnn.d_0m0);
#ifdef P4_TO_P8
    double s_00p = fabs(qnnn.d_00p); double s_00m = fabs(qnnn.d_00m);
#endif
#ifdef P4_TO_P8
    double s_min = MIN(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0), MIN(s_00p, s_00m));
#else
    double s_min = MIN(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0));
#endif

    /* choose CFL = 0.8 ... just for fun! */
    dt_local = MIN(dt_local, 0.8*fabs(s_min/vn_p[n]));
  }
  MPI_Allreduce(&dt_local, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm);

  Vec p1, p2;
  double *p1_p, *p2_p;
  ierr = VecDuplicate(phi_xx_, &p1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_yy_, &p2); CHKERRXX(ierr);

  ierr = VecGetArray(p1, &p1_p); CHKERRXX(ierr);
  ierr = VecGetArray(p2, &p2_p); CHKERRXX(ierr);

  /* p1(Ln, Gn, Gn) */
  memcpy(p1_p, phi_p, sizeof(double) * nodes->indep_nodes.elem_count);

  // layer nodes
  advect_in_normal_direction_one_iteration(ngbd->layer_nodes, vn_p, dt,
                                         #ifdef P4_TO_P8
                                           phi_xx_p, phi_yy_p, phi_zz_p,
                                         #else
                                           phi_xx_p, phi_yy_p,
                                         #endif
                                           p1_p, phi_p);
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // local nodes
  advect_in_normal_direction_one_iteration(ngbd->local_nodes, vn_p, dt,
                                         #ifdef P4_TO_P8
                                           phi_xx_p, phi_yy_p, phi_zz_p,
                                         #else
                                           phi_xx_p, phi_yy_p,
                                         #endif
                                           p1_p, phi_p);
  ierr = VecGhostUpdateEnd  (phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* now phi(Lnp1, Bnp1, Gnp1) */
#ifdef P4_TO_P8
  compute_derivatives(phi, phi_xx_, phi_yy_, phi_zz_);
#else
  compute_derivatives(phi, phi_xx_, phi_yy_);
#endif
  // Advect the function in time and then get a computed timestep

  // layer nodes
  advect_in_normal_direction_one_iteration(ngbd->layer_nodes, vn_p, dt,
                                         #ifdef P4_TO_P8
                                           phi_xx_p, phi_yy_p, phi_zz_p,
                                         #else
                                           phi_xx_p, phi_yy_p,
                                         #endif
                                           phi_p, p2_p);
  ierr = VecGhostUpdateBegin(p2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // local nodes
  advect_in_normal_direction_one_iteration(ngbd->local_nodes, vn_p, dt,
                                         #ifdef P4_TO_P8
                                           phi_xx_p, phi_yy_p, phi_zz_p,
                                         #else
                                           phi_xx_p, phi_yy_p,
                                         #endif
                                           phi_p, p2_p);
  ierr = VecGhostUpdateEnd  (p2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    phi_p[n] = 0.5 * (p1_p[n] + p2_p[n]);

  /* restore arrays */
  ierr = VecRestoreArray(phi,     &phi_p);    CHKERRXX(ierr);
  ierr = VecRestoreArray(vn,      &vn_p);     CHKERRXX(ierr);
  ierr = VecRestoreArray(p1,      &p1_p);     CHKERRXX(ierr);
  ierr = VecRestoreArray(p2,      &p2_p);     CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif
  ierr = PetscLogFlops(nodes->num_owned_indeps * P4EST_DIM); CHKERRXX(ierr);

  if (local_derivatives){
    ierr = VecDestroy(phi_xx_); CHKERRXX(ierr);
    ierr = VecDestroy(phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(phi_zz_); CHKERRXX(ierr);
#endif
  }

  ierr = VecDestroy(p1); CHKERRXX(ierr);
  ierr = VecDestroy(p2); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_advect_in_normal_direction_Vec, 0, 0, 0, 0);

  return dt;
}

#ifdef P4_TO_P8
void my_p4est_level_set_t::extend_Over_Interface( Vec phi_petsc, Vec q_petsc, BoundaryConditions3D &bc, int order, int band_to_extend ) const
#else
void my_p4est_level_set_t::extend_Over_Interface( Vec phi_petsc, Vec q_petsc, BoundaryConditions2D &bc, int order, int band_to_extend ) const
#endif
{
#ifdef CASL_THROWS
  if(bc.interfaceType()==NOINTERFACE) throw std::invalid_argument("[CASL_ERROR]: extend_over_interface: no interface defined in the boundary condition ... needs to be dirichlet or neumann.");
  if(order!=0 && order!=1 && order!=2) throw std::invalid_argument("[CASL_ERROR]: extend_over_interface: invalid order. Choose 0, 1 or 2");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_over_interface, phi_petsc, q_petsc, 0, 0); CHKERRXX(ierr);

  double *phi;
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

  /* first compute the phi derivatives */
  std::vector<double> phi_x(nodes->num_owned_indeps);
  std::vector<double> phi_y(nodes->num_owned_indeps);
#ifdef P4_TO_P8
  std::vector<double> phi_z(nodes->num_owned_indeps);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);

    phi_x[n] = qnnn.dx_central(phi);
    phi_y[n] = qnnn.dy_central(phi);
#ifdef P4_TO_P8
    phi_z[n] = qnnn.dz_central(phi);
#endif
  }

  my_p4est_interpolation_nodes_t interp1(ngbd); interp1.set_input(q_petsc, quadratic_non_oscillatory);
  my_p4est_interpolation_nodes_t interp2(ngbd); interp2.set_input(q_petsc, quadratic_non_oscillatory);

  /* find dx and dy smallest */
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  double dy = (ymax-ymin) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax-zmin) / pow(2.,(double) data->max_lvl);
#endif

#ifdef P4_TO_P8
  double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
  double diag = sqrt(dx*dx + dy*dy);
#endif

  std::vector<double> q0;
  std::vector<double> q1;
  std::vector<double> q2;

  if(bc.interfaceType()==DIRICHLET)                           q0.resize(nodes->num_owned_indeps);
  if(order >= 1 || (order==0 && bc.interfaceType()==NEUMANN)) q1.resize(nodes->num_owned_indeps);
  if(order >= 2)                                              q2.resize(nodes->num_owned_indeps);

  /* now buffer the interpolation points */
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    Point3 grad_phi(-phi_x[n], -phi_y[n], -phi_z[n]);
#else
    Point2 grad_phi(-phi_x[n], -phi_y[n]);
#endif

    if(phi[n]>0 && phi[n]<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      grad_phi /= grad_phi.norm_L2();

      double xyz [P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      if(bc.interfaceType()==DIRICHLET)
#ifdef P4_TO_P8
        q0[n] = bc.interfaceValue(xyz[0] + grad_phi.x*phi[n], xyz[1] + grad_phi.y*phi[n], xyz[2] + grad_phi.z*phi[n]);
#else
        q0[n] = bc.interfaceValue(xyz[0] + grad_phi.x*phi[n], xyz[1] + grad_phi.y*phi[n]);
#endif

      if(order >= 1 || (order==0 && bc.interfaceType()==NEUMANN))
      {
        double xyz_ [] =
        {
          xyz[0] + grad_phi.x * (2*diag + phi[n]),
          xyz[1] + grad_phi.y * (2*diag + phi[n])
  #ifdef P4_TO_P8
          ,
          xyz[2] + grad_phi.z * (2*diag + phi[n])
  #endif
        };
        interp1.add_point(n, xyz_);
      }

      if(order >= 2)
      {
        double xyz_ [] =
        {
          xyz[0] + grad_phi.x * (3*diag + phi[n]),
          xyz[1] + grad_phi.y * (3*diag + phi[n])
  #ifdef P4_TO_P8
          ,
          xyz[2] + grad_phi.z * (3*diag + phi[n])
  #endif
        };
        interp2.add_point(n, xyz_);
      }

      ierr = PetscLogFlops(26); CHKERRXX(ierr);
    }
  }

  interp1.interpolate(q1.data());
  interp2.interpolate(q2.data());

  /* now compute the extrapolated values */
  double *q;
  ierr = VecGetArray(q_petsc, &q); CHKERRXX(ierr);
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    Point3 grad_phi(phi_x[n], phi_y[n], phi_z[n]);
#else
    Point2 grad_phi(phi_x[n], phi_y[n]);
#endif

    if(phi[n]>0 && phi[n]<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      grad_phi /= grad_phi.norm_L2();

      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      //      double xy1 [] = {xyz[0] - grad_phi.x*(phi[n]), xyz[1] - grad_phi.y*(phi[n])};
      //      double xy2 [] = {xyz[0] - grad_phi.x*(phi[n]+2*diag), xyz[1] - grad_phi.y*(phi[n]+2*diag)};

      //      q1[n] = 0.25*(
      //          bc.interfaceValue(xyz[0] - grad_phi.x*(phi[n]+2*diag), xyz[1] - grad_phi.y*(phi[n]+2*diag)) +
      //          bc.interfaceValue(xyz[0] - grad_phi.x*(phi[n]+2*diag), xyz[1] - grad_phi.y*(phi[n]+2*diag)) +
      //          bc.interfaceValue(xyz[0] - grad_phi.x*(phi[n]+2*diag), xyz[1] - grad_phi.y*(phi[n]+2*diag)) +
      //          bc.interfaceValue(xyz[0] - grad_phi.x*(phi[n]+2*diag), xyz[1] - grad_phi.y*(phi[n]+2*diag)) );
      //      q2[n] = bc.interfaceValue(xyz[0] - grad_phi.x*(phi[n]+3*diag), xyz[1] - grad_phi.y*(phi[n]+3*diag));

      if(order==0)
      {
        if(bc.interfaceType()==DIRICHLET)
          q[n] = q0[n];
        else /* interface neumann */
          q[n] = q1[n];
      }

      else if(order==1)
      {
        if(bc.interfaceType()==DIRICHLET)
        {
          double dif01 = (q1[n] - q0[n])/(2*diag - 0);
          //          double dif01 = (q1[n] - q0[n])/(sqrt(SQR(xy2[0]-xy1[0])+SQR(xy2[1]-xy1[1])) - 0);
          q[n] = q0[n] + (-phi[n] - 0) * dif01;
        }
        else /* interface Neumann */
        {
#ifdef P4_TO_P8
          double dif01 = -bc.interfaceValue(xyz[0]-grad_phi.x*phi[n], xyz[1]-grad_phi.y*phi[n], xyz[2]-grad_phi.z*phi[n]);
#else
          double dif01 = -bc.interfaceValue(xyz[0]-grad_phi.x*phi[n], xyz[1]-grad_phi.y*phi[n]);
#endif
          q[n] = q1[n] + (-phi[n] - 2*diag) * dif01;
        }
      }

      else if(order==2)
      {
        if(bc.interfaceType()==DIRICHLET)
        {
          double dif01  = (q1[n] - q0[n]) / (2*diag);
          double dif12  = (q2[n] - q1[n]) / (diag);
          double dif012 = (dif12 - dif01) / (3*diag);
          q[n] = q0[n] + (-phi[n] - 0) * dif01 + (-phi[n] - 0)*(-phi[n] - 2*diag) * dif012;
        }
        else if (bc.interfaceType() == NEUMANN) /* interface Neumann */
        {
          double x1 = 2*diag;
          double x2 = 3*diag;
          double b = -bc.interfaceValue(xyz[0]-grad_phi.x*phi[n], xyz[1]-grad_phi.y*phi[n]
    #ifdef P4_TO_P8
              , xyz[2]-grad_phi.z*phi[n]
    #endif
              );
          double a = (q2[n] - q1[n] + b*(x1 - x2)) / (x2*x2 - x1*x1);
          double c = q1[n] - a*x1*x1 - b*x1;

          double x = -phi[n];
          q[n] = a*x*x + b*x + c;

          //          double dif01 = (q2[n] - q1[n])/(diag);
          //#ifdef P4_TO_P8
          //          double dif012 = (dif01 + bc.interfaceValue(xyz[0]-grad_phi.x*phi[n], xyz[1]-grad_phi.y*phi[n], xyz[2]-grad_phi.z*phi[n])) / (2*diag);
          //#else
          //          double dif012 = (dif01 + bc.interfaceValue(xyz[0]-grad_phi.x*phi[n], xyz[1]-grad_phi.y*phi[n])) / (2*diag);
          //#endif
          //          q[n] = q1[n] + (-phi[n] - diag) * dif01 + (-phi[n] - diag)*(-phi[n] - 2*diag) * dif012;
        }
      }

      ierr = PetscLogFlops(48); CHKERRXX(ierr);
    }
  }
  ierr = VecRestoreArray(q_petsc, &q); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(q_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (q_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_over_interface, phi_petsc, q_petsc, 0, 0); CHKERRXX(ierr);
}

void my_p4est_level_set_t::extend_Over_Interface( Vec phi_petsc, Vec q_petsc, BoundaryConditionType bc_type, Vec bc_vec, int order, int band_to_extend ) const
{

#ifdef CASL_THROWS
  if(bc_type==NOINTERFACE) throw std::invalid_argument("[CASL_ERROR]: extend_over_interface: no interface defined in the boundary condition ... needs to be dirichlet or neumann.");
  if(order!=0 && order!=1 && order!=2) throw std::invalid_argument("[CASL_ERROR]: extend_over_interface: invalid order. Choose 0, 1 or 2");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_over_interface, phi_petsc, q_petsc, 0, 0); CHKERRXX(ierr);

  double *phi;
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

  /* first compute the phi derivatives */
  std::vector<double> phi_x(nodes->num_owned_indeps);
  std::vector<double> phi_y(nodes->num_owned_indeps);
#ifdef P4_TO_P8
  std::vector<double> phi_z(nodes->num_owned_indeps);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);

    phi_x[n] = qnnn.dx_central(phi);
    phi_y[n] = qnnn.dy_central(phi);
#ifdef P4_TO_P8
    phi_z[n] = qnnn.dz_central(phi);
#endif
  }

  my_p4est_interpolation_nodes_t interp0(ngbd); interp0.set_input(q_petsc, quadratic_non_oscillatory);
  my_p4est_interpolation_nodes_t interp1(ngbd); interp1.set_input(q_petsc, quadratic_non_oscillatory);
  my_p4est_interpolation_nodes_t interp2(ngbd); interp2.set_input(q_petsc, quadratic_non_oscillatory);

  /* find dx and dy smallest */
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  double dy = (ymax-ymin) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax-zmin) / pow(2.,(double) data->max_lvl);
#endif

#ifdef P4_TO_P8
  double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
  double diag = sqrt(dx*dx + dy*dy);
#endif

  std::vector<double> q0;
  std::vector<double> q1;
  std::vector<double> q2;

  if(order > 0 || bc_type==DIRICHLET)              q0.resize(nodes->num_owned_indeps);
  if(order >= 1 || (order==0 && bc_type==NEUMANN)) q1.resize(nodes->num_owned_indeps);
  if(order >= 2)                                   q2.resize(nodes->num_owned_indeps);

  /* now buffer the interpolation points */
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    Point3 grad_phi(-phi_x[n], -phi_y[n], -phi_z[n]);
#else
    Point2 grad_phi(-phi_x[n], -phi_y[n]);
#endif

    if(phi[n]>0 && phi[n]<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      grad_phi /= grad_phi.norm_L2();

      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      if(order>0 || bc_type==DIRICHLET){
        double xyz_ [] =
        {
          xyz[0] + grad_phi.x * phi[n],
          xyz[1] + grad_phi.y * phi[n]
  #ifdef P4_TO_P8
          ,
          xyz[2] + grad_phi.z * phi[n]
  #endif
        };

        interp0.add_point(n, xyz_);
      }

      if(order >= 1 || (order==0 && bc_type==NEUMANN))
      {
        double xyz_ [] =
        {
          xyz[0] + grad_phi.x * (2*diag + phi[n]),
          xyz[1] + grad_phi.y * (2*diag + phi[n])
  #ifdef P4_TO_P8
          ,
          xyz[2] + grad_phi.z * (2*diag + phi[n])
  #endif
        };
        interp1.add_point(n, xyz_);
      }

      if(order >= 2)
      {
        double xyz_ [] =
        {
          xyz[0] + grad_phi.x * (3*diag + phi[n]),
          xyz[1] + grad_phi.y * (3*diag + phi[n])
  #ifdef P4_TO_P8
          ,
          xyz[2] + grad_phi.z * (3*diag + phi[n])
  #endif
        };
        interp2.add_point(n, xyz_);
      }

      ierr = PetscLogFlops(26); CHKERRXX(ierr);
    }
  }

  interp0.interpolate(q0.data());
  interp1.interpolate(q1.data());
  interp2.interpolate(q2.data());

  /* now compute the extrapolated values */
  double *q;
  ierr = VecGetArray(q_petsc, &q); CHKERRXX(ierr);
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    Point3 grad_phi(phi_x[n], phi_y[n], phi_z[n]);
#else
    Point2 grad_phi(phi_x[n], phi_y[n]);
#endif

    if(phi[n]>0 && phi[n]<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      if(order==0)
      {
        if(bc_type==DIRICHLET)
          q[n] = q0[n];
        else /* interface neumann */
          q[n] = q1[n];
      }

      else if(order==1)
      {
        if(bc_type==DIRICHLET)
        {
          double dif01 = (q1[n] - q0[n])/(2*diag - 0);
          q[n] = q0[n] + (-phi[n] - 0) * dif01;
        }
        else /* interface Neumann */
        {
          double dif01 = -q0[n];
          q[n] = q1[n] + (-phi[n] - 2*diag) * dif01;
        }
      }

      else if(order==2)
      {
        if(bc_type==DIRICHLET)
        {
          double dif01  = (q1[n] - q0[n]) / (2*diag);
          double dif12  = (q2[n] - q1[n]) / (diag);
          double dif012 = (dif12 - dif01) / (3*diag);
          q[n] = q0[n] + (-phi[n] - 0) * dif01 + (-phi[n] - 0)*(-phi[n] - 2*diag) * dif012;
        }
        else /* interface Neumann */
        {
          double x1 = 2*diag;
          double x2 = 3*diag;
          double b = -q0[n];
          double a = (q2[n] - q1[n] + b*(x1 - x2)) / (x2*x2 - x1*x1);
          double c = q1[n] - a*x1*x1 - b*x1;

          double x = -phi[n];
          q[n] = a*x*x + b*x + c;
        }
      }

      ierr = PetscLogFlops(48); CHKERRXX(ierr);
    }
  }
  ierr = VecRestoreArray(q_petsc, &q); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(q_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (q_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_over_interface, phi_petsc, q_petsc, 0, 0); CHKERRXX(ierr);
}

void my_p4est_level_set_t::extend_Over_Interface(Vec phi_petsc, Vec q_petsc, int order, int band_to_extend ) const
{
#ifdef CASL_THROWS
  if(order!=0 && order!=1 && order!=2) throw std::invalid_argument("[CASL_ERROR]: extend_over_interface: invalid order. Choose 0, 1 or 2");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_over_interface, phi_petsc, q_petsc, 0, 0); CHKERRXX(ierr);

  double *phi;
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

  /* first compute the phi derivatives */
  std::vector<double> phi_x(nodes->num_owned_indeps);
  std::vector<double> phi_y(nodes->num_owned_indeps);
#ifdef P4_TO_P8
  std::vector<double> phi_z(nodes->num_owned_indeps);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);

    phi_x[n] = qnnn.dx_central(phi);
    phi_y[n] = qnnn.dy_central(phi);
#ifdef P4_TO_P8
    phi_z[n] = qnnn.dz_central(phi);
#endif
  }

  my_p4est_interpolation_nodes_t interp0(ngbd); interp0.set_input(q_petsc, quadratic_non_oscillatory);
  my_p4est_interpolation_nodes_t interp1(ngbd); interp1.set_input(q_petsc, quadratic_non_oscillatory);
  my_p4est_interpolation_nodes_t interp2(ngbd); interp2.set_input(q_petsc, quadratic_non_oscillatory);

  /* find dx and dy smallest */
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  double dy = (ymax-ymin) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax-zmin) / pow(2.,(double) data->max_lvl);
#endif

#ifdef P4_TO_P8
  double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
  double diag = sqrt(dx*dx + dy*dy);
#endif

  std::vector<double> q0;
  std::vector<double> q1;
  std::vector<double> q2;

  if(order >  0) q0.resize(nodes->num_owned_indeps);
  if(order >= 1) q1.resize(nodes->num_owned_indeps);
  if(order >= 2) q2.resize(nodes->num_owned_indeps);

  /* now buffer the interpolation points */
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    Point3 grad_phi(-phi_x[n], -phi_y[n], -phi_z[n]);
#else
    Point2 grad_phi(-phi_x[n], -phi_y[n]);
#endif

    if(phi[n]>0 && phi[n]<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      grad_phi /= grad_phi.norm_L2();

      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      if(order>0){
        double xyz_ [] =
        {
          xyz[0] + grad_phi.x * (2*diag + phi[n]),
          xyz[1] + grad_phi.y * (2*diag + phi[n])
  #ifdef P4_TO_P8
          ,
          xyz[2] + grad_phi.z * (2*diag + phi[n])
  #endif
        };

        interp0.add_point(n, xyz_);
      }

      if(order >= 1)
      {
        double xyz_ [] =
        {
          xyz[0] + grad_phi.x * (3*diag + phi[n]),
          xyz[1] + grad_phi.y * (3*diag + phi[n])
  #ifdef P4_TO_P8
          ,
          xyz[2] + grad_phi.z * (3*diag + phi[n])
  #endif
        };
        interp1.add_point(n, xyz_);
      }

      if(order >= 2)
      {
        double xyz_ [] =
        {
          xyz[0] + grad_phi.x * (4*diag + phi[n]),
          xyz[1] + grad_phi.y * (4*diag + phi[n])
  #ifdef P4_TO_P8
          ,
          xyz[2] + grad_phi.z * (4*diag + phi[n])
  #endif
        };
        interp2.add_point(n, xyz_);
      }

      ierr = PetscLogFlops(26); CHKERRXX(ierr);
    }
  }

  interp0.interpolate(q0.data());
  interp1.interpolate(q1.data());
  interp2.interpolate(q2.data());

  /* now compute the extrapolated values */
  double *q;
  ierr = VecGetArray(q_petsc, &q); CHKERRXX(ierr);
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    Point3 grad_phi(phi_x[n], phi_y[n], phi_z[n]);
#else
    Point2 grad_phi(phi_x[n], phi_y[n]);
#endif

    if(phi[n]>0 && phi[n]<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      if(order==0)
      {
        q[n] = q0[n];
      }

      else if(order==1)
      {
        double dif01 = (q1[n] - q0[n])/(diag);
        q[n] = q0[n] + (-phi[n] - 2*diag) * dif01;
      }

      else if(order==2)
      {
        double dif01  = (q1[n] - q0[n]) / (diag);
        double dif12  = (q2[n] - q1[n]) / (diag);
        double dif012 = (dif12 - dif01) / (2*diag);
        q[n] = q0[n] + (-phi[n] - 2*diag) * dif01 + (-phi[n] - 2*diag)*(-phi[n] - 3*diag) * dif012;
      }

      ierr = PetscLogFlops(48); CHKERRXX(ierr);
    }
  }
  ierr = VecRestoreArray(q_petsc, &q); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(q_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (q_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_over_interface, phi_petsc, q_petsc, 0, 0); CHKERRXX(ierr);
}

void my_p4est_level_set_t::extend_from_interface_to_whole_domain( Vec phi_petsc, Vec q_petsc, Vec q_extended_petsc, int band_to_extend) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_from_interface, phi_petsc, q_petsc, q_extended_petsc, 0); CHKERRXX(ierr);

  /* find dx and dy smallest */
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;

  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax-xmin) / (1<<data->max_lvl);
  double dy = (ymax-ymin) / (1<<data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax-zmin) / (1<<data->max_lvl);
#endif

#ifdef P4_TO_P8
  double diag = sqrt(dx*dx + dy*dy + dz*dz);
#else
  double diag = sqrt(dx*dx + dy*dy);
#endif

  my_p4est_interpolation_nodes_t interp(ngbd);

  double *q_extended;
  ierr = VecGetArray(q_extended_petsc, &q_extended); CHKERRXX(ierr);

  double *phi;
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

  /* now buffer the interpolation points */
  quad_neighbor_nodes_of_node_t qnnn;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);

#ifdef P4_TO_P8
    Point3 grad_phi;
#else
    Point2 grad_phi;
#endif
    grad_phi.x = qnnn.dx_central(phi);
    grad_phi.y = qnnn.dy_central(phi);
#ifdef P4_TO_P8
    grad_phi.z = qnnn.dz_central(phi);
#endif

    if(fabs(phi[n])<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      grad_phi /= grad_phi.norm_L2();

      double xyz[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz);

      interp.add_point(n, xyz);

      ierr = PetscLogFlops(14); CHKERRXX(ierr);
    }
    else
      q_extended[n] = 0;
  }

  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);
  ierr = VecRestoreArray(q_extended_petsc, &q_extended); CHKERRXX(ierr);

  interp.set_input(q_petsc, quadratic);
  interp.interpolate(q_extended_petsc);

  ierr = VecGhostUpdateBegin(q_extended_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (q_extended_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_from_interface, phi_petsc, q_petsc, q_extended_petsc, 0); CHKERRXX(ierr);
}


void my_p4est_level_set_t::extend_Over_Interface_TVD(Vec phi, Vec q, int iterations, int order,
                                                     double tol, double band_use, double band_extend, double band_check,
                                                     Vec normal[], Vec mask, boundary_conditions_t *bc,
                                                     bool use_nonzero_guess, Vec q_n, Vec q_nn) const
{
#ifdef CASL_THROWS
  if(order!=0 && order!=1 && order!=2) throw std::invalid_argument("[CASL_ERROR]: my_p4est_level_set_t->extend_Over_Interface_TVD: order must be 0, 1 or 2.");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_over_interface_TVD, phi, q, 0, 0); CHKERRXX(ierr);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  double *mask_p;
  if (mask != NULL) {
    ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);
  } else {
    mask_p = phi_p;
  }

  if (band_use > 0) band_use = -band_use;

  Vec qn, qnn;
  double *q_p, *qn_p, *qnn_p;
  Vec b_qn_well_defined;
  Vec b_qnn_well_defined;
  double *b_qn_well_defined_p;
  double *b_qnn_well_defined_p;

  Vec tmp, tmp_loc;
  Vec qnn_loc, qn_loc, q_loc;
  ierr = VecDuplicate(phi, &tmp); CHKERRXX(ierr);
  double *tmp_p;

  double dxyz[P4EST_DIM];

  dxyz_min(p4est, dxyz);

#ifdef P4_TO_P8
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
#else
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
#endif

  double rel_thresh = 1.e-2;

  double tol_d  = tol/diag;
  double tol_dd = tol_d/diag;

  /* init the neighborhood information if needed */
  /* NOTE: from now on the neighbors will be initialized ... do we want to clear them
   * at the end of this function if they were not initialized beforehand ?
   */
  ngbd->init_neighbors();

  /* compute the normals */
  double *nx, *ny, *nz;

  if (normal != NULL)
  {
    ierr = VecGetArray(normal[0], &nx); CHKERRXX(ierr);
    ierr = VecGetArray(normal[1], &ny); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(normal[2], &nz); CHKERRXX(ierr);
#endif
  } else {
    nx = new double[nodes->num_owned_indeps];
    ny = new double[nodes->num_owned_indeps];
#ifdef P4_TO_P8
    nz = new double[nodes->num_owned_indeps];
#endif

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if (phi_p[n] > band_use && phi_p[n] < band_extend)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        nx[n] = qnnn.dx_central(phi_p);
        ny[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
        nz[n] = qnnn.dz_central(phi_p);
        double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n] + nz[n]*nz[n]);
#else
        double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n]);
#endif

        if(norm>EPS)
        {
          nx[n] /= norm;
          ny[n] /= norm;
#ifdef P4_TO_P8
          nz[n] /= norm;
#endif
        }
        else
        {
          nx[n] = 0;
          ny[n] = 0;
#ifdef P4_TO_P8
          nz[n] = 0;
#endif
        }
      }
    }
  }


  ierr = VecGetArray(q , &q_p) ; CHKERRXX(ierr);

  /* initialize qn */
  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  if(order >=1 )
  {
    // first-order derivatives
    if (q_n == NULL) { ierr = VecCreateGhostNodes(p4est, nodes, &qn); CHKERRXX(ierr); }
    else qn = q_n;

    ierr = VecCreateGhostNodes(p4est, nodes, &b_qn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      if (phi_p[n] > band_use)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        if(  mask_p[qnnn.node_000]<-EPS &&
     #ifdef P4_TO_P8
             ( mask_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( mask_p[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
             ( mask_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( mask_p[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

             ( mask_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( mask_p[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
             ( mask_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( mask_p[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

             ( mask_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( mask_p[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
             ( mask_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( mask_p[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

             ( mask_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( mask_p[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
             ( mask_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( mask_p[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

             ( mask_p[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( mask_p[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
             ( mask_p[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( mask_p[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

             ( mask_p[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( mask_p[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
             ( mask_p[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( mask_p[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
             ( mask_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
             ( mask_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
             ( mask_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
             ( mask_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
             ( mask_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
             ( mask_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
             ( mask_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
             ( mask_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
             )
        {
          b_qn_well_defined_p[n] = true;
          qn_p[n] = ( nx[n]*qnnn.dx_central(q_p) +
                      ny[n]*qnnn.dy_central(q_p)
            #ifdef P4_TO_P8
                      + nz[n]*qnnn.dz_central(q_p)
            #endif
                      );
        }
        else if (mask_p[qnnn.node_000]<-EPS && bc != NULL)
        {
          b_qn_well_defined_p[n] = false;
          qn_p[n] = use_nonzero_guess ? SUMD( nx[n]*qnnn.dx_central(q_p),
                                              ny[n]*qnnn.dy_central(q_p),
                                              nz[n]*qnnn.dz_central(q_p) ) : 0;

          if (bc->type == DIRICHLET && bc->pointwise && bc->num_value_pts(n) < P4EST_DIM+1 && bc->num_value_pts(n) > 0)
          {
            double d_m00 = qnnn.d_m00, d_p00 = qnnn.d_p00;
            double d_0m0 = qnnn.d_0m0, d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
            double d_00m = qnnn.d_00m, d_00p = qnnn.d_00p;
#endif

            // assuming grid is uniform near the interface
            double q_m00 = qnnn.f_m00_linear(q_p), q_p00 = qnnn.f_p00_linear(q_p);
            double q_0m0 = qnnn.f_0m0_linear(q_p), q_0p0 = qnnn.f_0p0_linear(q_p);
#ifdef P4_TO_P8
            double q_00m = qnnn.f_00m_linear(q_p), q_00p = qnnn.f_00p_linear(q_p);
#endif
            double q_000 = q_p[n];

            double d_min = diag;
            for (unsigned int i = 0; i < bc->num_value_pts(n); ++i)
            {
              int idx = bc->idx_value_pt(n,i);
              interface_point_cartesian_t *pt = &bc->dirichlet_pts[idx];
              switch (pt->dir)
              {
                case 0: d_m00 = pt->dist; q_m00 = bc->get_value_pw(n,i); break;
                case 1: d_p00 = pt->dist; q_p00 = bc->get_value_pw(n,i); break;
                case 2: d_0m0 = pt->dist; q_0m0 = bc->get_value_pw(n,i); break;
                case 3: d_0p0 = pt->dist; q_0p0 = bc->get_value_pw(n,i); break;
#ifdef P4_TO_P8
                case 4: d_00m = pt->dist; q_00m = bc->get_value_pw(n,i); break;
                case 5: d_00p = pt->dist; q_00p = bc->get_value_pw(n,i); break;
#endif
              }

              d_min = MIN(d_min, pt->dist);
            }

            if (d_min > rel_thresh*diag)
            {
              b_qn_well_defined_p[n] = true;

              qn_p[n] = ( nx[n]*((q_p00-q_000)*d_m00/d_p00 + (q_000-q_m00)*d_p00/d_m00)/(d_m00+d_p00) +
                          ny[n]*((q_0p0-q_000)*d_0m0/d_0p0 + (q_000-q_0m0)*d_0p0/d_0m0)/(d_0m0+d_0p0)
            #ifdef P4_TO_P8
                          + nz[n]*((q_00p-q_000)*d_00m/d_00p + (q_000-q_00m)*d_00p/d_00m)/(d_00m+d_00p)
            #endif
                          );
            }
          }
        }
        else
        {
          b_qn_well_defined_p[n] = false;
          qn_p[n] = use_nonzero_guess ? SUMD( nx[n]*qnnn.dx_central(q_p),
                                              ny[n]*qnnn.dy_central(q_p),
                                              nz[n]*qnnn.dz_central(q_p) ) : 0;
        }
      }
      else
      {
        b_qn_well_defined_p[n] = true;
        qn_p[n] = 0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      if (phi_p[n] > band_use)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        if(  mask_p[qnnn.node_000]<-EPS &&
     #ifdef P4_TO_P8
             ( mask_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( mask_p[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
             ( mask_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( mask_p[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

             ( mask_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( mask_p[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
             ( mask_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( mask_p[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

             ( mask_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( mask_p[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
             ( mask_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( mask_p[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

             ( mask_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( mask_p[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
             ( mask_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( mask_p[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

             ( mask_p[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( mask_p[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
             ( mask_p[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( mask_p[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

             ( mask_p[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( mask_p[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
             ( mask_p[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( mask_p[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
             ( mask_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
             ( mask_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
             ( mask_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
             ( mask_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
             ( mask_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
             ( mask_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
             ( mask_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
             ( mask_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
             )
        {
          b_qn_well_defined_p[n] = true;
          qn_p[n] = ( nx[n]*qnnn.dx_central(q_p) +
                      ny[n]*qnnn.dy_central(q_p)
            #ifdef P4_TO_P8
                      + nz[n]*qnnn.dz_central(q_p)
            #endif
                      );
        }
        else if (mask_p[qnnn.node_000]<-EPS && bc != NULL)
        {
          b_qn_well_defined_p[n] = false;
          qn_p[n] = use_nonzero_guess ? SUMD( nx[n]*qnnn.dx_central(q_p),
                                              ny[n]*qnnn.dy_central(q_p),
                                              nz[n]*qnnn.dz_central(q_p) ) : 0;

          if (bc->type == DIRICHLET && bc->pointwise && bc->num_value_pts(n) < P4EST_DIM+1 && bc->num_value_pts(n) > 0)
          {
            double d_m00 = qnnn.d_m00, d_p00 = qnnn.d_p00;
            double d_0m0 = qnnn.d_0m0, d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
            double d_00m = qnnn.d_00m, d_00p = qnnn.d_00p;
#endif

            // assuming grid is uniform near the interface
            double q_m00 = qnnn.f_m00_linear(q_p), q_p00 = qnnn.f_p00_linear(q_p);
            double q_0m0 = qnnn.f_0m0_linear(q_p), q_0p0 = qnnn.f_0p0_linear(q_p);
#ifdef P4_TO_P8
            double q_00m = qnnn.f_00m_linear(q_p), q_00p = qnnn.f_00p_linear(q_p);
#endif
            double q_000 = q_p[n];

            double d_min = diag;
            for (unsigned int i = 0; i < bc->num_value_pts(n); ++i)
            {
              int idx = bc->idx_value_pt(n,i);
              interface_point_cartesian_t *pt = &bc->dirichlet_pts[idx];
              switch (pt->dir)
              {
                case 0: d_m00 = pt->dist; q_m00 = bc->get_value_pw(n,i); break;
                case 1: d_p00 = pt->dist; q_p00 = bc->get_value_pw(n,i); break;
                case 2: d_0m0 = pt->dist; q_0m0 = bc->get_value_pw(n,i); break;
                case 3: d_0p0 = pt->dist; q_0p0 = bc->get_value_pw(n,i); break;
#ifdef P4_TO_P8
                case 4: d_00m = pt->dist; q_00m = bc->get_value_pw(n,i); break;
                case 5: d_00p = pt->dist; q_00p = bc->get_value_pw(n,i); break;
#endif
              }

              d_min = MIN(d_min, pt->dist);
            }

            if (d_min > rel_thresh*diag)
            {
              b_qn_well_defined_p[n] = true;

              qn_p[n] = ( nx[n]*((q_p00-q_000)*d_m00/d_p00 + (q_000-q_m00)*d_p00/d_m00)/(d_m00+d_p00) +
                          ny[n]*((q_0p0-q_000)*d_0m0/d_0p0 + (q_000-q_0m0)*d_0p0/d_0m0)/(d_0m0+d_0p0)
            #ifdef P4_TO_P8
                          + nz[n]*((q_00p-q_000)*d_00m/d_00p + (q_000-q_00m)*d_00p/d_00m)/(d_00m+d_00p)
            #endif
                          );
            }
          }
        }
        else
        {
          b_qn_well_defined_p[n] = false;
          qn_p[n] = use_nonzero_guess ? SUMD( nx[n]*qnnn.dx_central(q_p),
                                              ny[n]*qnnn.dy_central(q_p),
                                              nz[n]*qnnn.dz_central(q_p) ) : 0;
        }
      }
      else
      {
        b_qn_well_defined_p[n] = true;
        qn_p[n] = 0;
      }
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);

  /* initialize qnn */
  if(order == 2)
  {

    if (q_nn == NULL) { ierr = VecDuplicate(qn, &qnn); CHKERRXX(ierr); }
    else { qnn = q_nn; }
    ierr = VecDuplicate(b_qn_well_defined, &b_qnn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecGetArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      if (phi_p[n] > band_use)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        if(  b_qn_well_defined_p[qnnn.node_000]==true &&
     #ifdef P4_TO_P8
             ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_m00_mp]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_m00_pp]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_mp]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_pp]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_mp]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_pp]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_mp]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_pp]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_00m_mm]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00m_mp]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00m_pm]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00m_pp]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_00p_mm]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00p_mp]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00p_pm]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00p_pp]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
             ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
             )


        {
          b_qnn_well_defined_p[n] = true;
          qnn_p[n] = ( nx[n]*qnnn.dx_central(qn_p) +
                       ny[n]*qnnn.dy_central(qn_p)
             #ifdef P4_TO_P8
                       + nz[n]*qnnn.dz_central(qn_p)
             #endif
                       );
        }
        else
        {
          b_qnn_well_defined_p[n] = false;
          qnn_p[n] = use_nonzero_guess ? SUMD( nx[n]*qnnn.dx_central(qn_p),
                                               ny[n]*qnnn.dy_central(qn_p),
                                               nz[n]*qnnn.dz_central(qn_p) ) : 0;
        }
      }
      else
      {
        b_qnn_well_defined_p[n] = true;
        qnn_p[n] = 0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      if (phi_p[n] > band_use)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        if(  b_qn_well_defined_p[qnnn.node_000]==true &&
     #ifdef P4_TO_P8
             ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_m00_mp]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_m00_pp]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_mp]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_pp]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_mp]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_pp]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_mp]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_pp]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_00m_mm]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00m_mp]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00m_pm]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00m_pp]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

             ( b_qn_well_defined_p[qnnn.node_00p_mm]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00p_mp]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00p_pm]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_00p_pp]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
             ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS) &&
             ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
             )


        {
          b_qnn_well_defined_p[n] = true;
          qnn_p[n] = ( nx[n]*qnnn.dx_central(qn_p) +
                       ny[n]*qnnn.dy_central(qn_p)
             #ifdef P4_TO_P8
                       + nz[n]*qnnn.dz_central(qn_p)
             #endif
                       );
        }
        else
        {
          b_qnn_well_defined_p[n] = false;
          qnn_p[n] = use_nonzero_guess ? SUMD( nx[n]*qnnn.dx_central(qn_p),
                                               ny[n]*qnnn.dy_central(qn_p),
                                               nz[n]*qnnn.dz_central(qn_p) ) : 0;
        }
      }
      else
      {
        b_qnn_well_defined_p[n] = true;
        qnn_p[n] = 0;
      }
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  /* extrapolate qnn */
  if(order==2)
  {
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    int    it     = 0;
    double change = tol_dd+1;
    while (it < iterations && change > tol_dd)
    {
      change = 0;
      ++it;
      ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qnn_well_defined_p[n] && phi_p[n] < band_extend)
        {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnnx = nx[n]>0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n]>0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n]>0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          double change_loc = dt*SUMD( nx[n]*qnnx, ny[n]*qnny, nz[n]*qnnz );
          tmp_p[n] = qnn_p[n] - change_loc;

          if (phi_p[n] < band_check)
          {
            if (fabs(change_loc) > change) change = fabs(change_loc);
          }
        }
        else
          tmp_p[n] = qnn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qnn_well_defined_p[n] && phi_p[n] < band_extend)
        {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnnx = nx[n]>0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n]>0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n]>0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          double change_loc = dt*SUMD( nx[n]*qnnx, ny[n]*qnny, nz[n]*qnnz );
          tmp_p[n] = qnn_p[n] - change_loc;

          if (phi_p[n] < band_check)
          {
            if (fabs(change_loc) > change) change = fabs(change_loc);
          }
        }
        else
          tmp_p[n] = qnn_p[n];
      }

      /* end update communication */
      ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      if (verbose)
      {
        PetscPrintf(p4est->mpicomm, "Extending second derivative, iteration: %d, error: %e\n", it, change);
      }

      ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

      ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(qnn, &qnn_loc); CHKERRXX(ierr);
      ierr = VecCopy(tmp_loc, qnn_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(qnn, &qnn_loc); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  /* extrapolate qn */
  if(order>=1)
  {
    if(order==2) ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    int    it     = 0;
    double change = tol+1;
    while (it < iterations && change > tol_d)
    {
      change = 0;
      ++it;
      ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qn_well_defined_p[n] && phi_p[n] < band_extend)
        {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnx = nx[n]>0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                               : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n]>0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                               : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n]>0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                               : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          double change_loc = dt*SUMD( nx[n]*qnx, ny[n]*qny, nz[n]*qnz ) - (order==2 ? dt*qnn_p[n] : 0);
          tmp_p[n] = qn_p[n] - change_loc;

          if (phi_p[n] < band_check)
          {
            if (fabs(change_loc) > change) change = fabs(change_loc);
          }
        }
        else
          tmp_p[n] = qn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qn_well_defined_p[n] && phi_p[n] < band_extend)
        {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnx = nx[n]>0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                               : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n]>0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                               : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n]>0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                               : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          double change_loc = dt*SUMD( nx[n]*qnx, ny[n]*qny, nz[n]*qnz ) - (order==2 ? dt*qnn_p[n] : 0);
          tmp_p[n] = qn_p[n] - change_loc;

          if (phi_p[n] < band_check)
          {
            if (fabs(change_loc) > change) change = fabs(change_loc);
          }
        }
        else
          tmp_p[n] = qn_p[n];
      }

      /* end update communication */
      ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);
      if (verbose)
      {
        PetscPrintf(p4est->mpicomm, "Extending first derivative, iteration: %d, error: %e\n", it, change);
      }

      ierr = VecRestoreArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

      ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(qn , &qn_loc ); CHKERRXX(ierr);
      ierr = VecCopy(tmp_loc, qn_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(qn , &qn_loc ); CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    if(order==2)
    {
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);
      if (q_nn == NULL) { ierr = VecDestroy(qnn); CHKERRXX(ierr); }
    }
  }

  if(order>=1) { ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr); }
  if(order==2) { ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr); }

  /* extrapolate q */
  Vec qxx, qyy;
  double *qxx_p, *qyy_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qxx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec qzz;
  double *qzz_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qzz); CHKERRXX(ierr);
#endif

  if(order>=1) { ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr); }

  int    it     = 0;
  double change = tol+1;
  while (it < iterations && change > tol)
  {
    change = 0;
    ++it;
#ifdef P4_TO_P8
    ngbd->second_derivatives_central(q, qxx, qyy, qzz);
#else
    ngbd->second_derivatives_central(q, qxx, qyy);
#endif

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      if(mask_p[n] > -EPS && phi_p[n] < band_extend)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
        dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
        dt /= 3;
#else
        dt /= 2;
#endif

        /* first order one sided derivatives */
        double qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                            : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                            : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                            : (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        /* second order derivatives */
        double qxx_m00 = qnnn.f_m00_linear(qxx_p);
        double qxx_p00 = qnnn.f_p00_linear(qxx_p);
        double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
        double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
        double qzz_00m = qnnn.f_00m_linear(qzz_p);
        double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

        /* minmod operation */
        qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
        else        qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
        else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

//#ifdef P4_TO_P8
//        if(fabs(nx[n])<EPS && fabs(ny[n])<EPS && fabs(nz[n])<EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p) +
//                      qnnn.f_00m_linear(q_p) + qnnn.f_00p_linear(q_p))/6.;
//#else
//        if(fabs(nx[n])<EPS && fabs(ny[n])<EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p))/4.;
//#endif
//        else
        double change_loc = dt*SUMD( nx[n]*qx, ny[n]*qy, nz[n]*qz ) - (order>=1 ? dt*qn_p[n] : 0);
        tmp_p[n] = q_p[n] - change_loc;

        if (phi_p[n] < band_check)
        {
          if (fabs(change_loc) > change) change = fabs(change_loc);
        }
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      if(mask_p[n] > -EPS && phi_p[n] < band_extend)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
        dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
        dt /= 3;
#else
        dt /= 2;
#endif

        /* first order one sided derivatives */
        double qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                            : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                            : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                            : (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        /* second order derivatives */
        double qxx_m00 = qnnn.f_m00_linear(qxx_p);
        double qxx_p00 = qnnn.f_p00_linear(qxx_p);
        double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
        double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
        double qzz_00m = qnnn.f_00m_linear(qzz_p);
        double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

        /* minmod operation */
        qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
        else        qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
        else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

//#ifdef P4_TO_P8
//        if(fabs(nx[n])<EPS && fabs(ny[n])<EPS && fabs(nz[n])<EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p) +
//                      qnnn.f_00m_linear(q_p) + qnnn.f_00p_linear(q_p))/6.;
//#else
//        if(fabs(nx[n])<EPS && fabs(ny[n])<EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p))/4.;
//#endif
//        else
        double change_loc = dt*SUMD( nx[n]*qx, ny[n]*qy, nz[n]*qz ) - (order>=1 ? dt*qn_p[n] : 0);
        tmp_p[n] = q_p[n] - change_loc;

        if (phi_p[n] < band_check)
        {
          if (fabs(change_loc) > change) change = fabs(change_loc);
        }
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    int mpiret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

    if (verbose)
    {
      PetscPrintf(p4est->mpicomm, "Extending values, iteration: %d, error: %e\n", it, change);
    }

    ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

    ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(q  , &q_loc  ); CHKERRXX(ierr);
    ierr = VecCopy(tmp_loc, q_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(q  , &q_loc  ); CHKERRXX(ierr);
  }

  if(order>=1)
  {
    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    if (q_n == NULL) { ierr = VecDestroy(qn); CHKERRXX(ierr); }
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  if (mask != NULL) {
    ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
  }

  ierr = VecDestroy(qxx); CHKERRXX(ierr);
  ierr = VecDestroy(qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(qzz); CHKERRXX(ierr);
#endif

  ierr = VecDestroy(tmp); CHKERRXX(ierr);

  if (normal != NULL)
  {
    ierr = VecRestoreArray(normal[0], &nx); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal[1], &ny); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(normal[2], &nz); CHKERRXX(ierr);
#endif
  } else {
    delete[] nx;
    delete[] ny;
#ifdef P4_TO_P8
    delete[] nz;
#endif
  }

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_over_interface_TVD, phi, q, 0, 0); CHKERRXX(ierr);
}


void my_p4est_level_set_t::extend_Over_Interface_TVD_Full(Vec phi, Vec q, int iterations, int order,
                                                          double tol, double band_use, double band_extend, double band_check,
                                                          Vec normal[P4EST_DIM], Vec mask, boundary_conditions_t *bc,
                                                          bool use_nonzero_guess, Vec *q_d, Vec *q_dd) const
{
#ifdef CASL_THROWS
  if(order!=0 && order!=1 && order!=2) throw std::invalid_argument("[CASL_ERROR]: my_p4est_level_set_t->extend_Over_Interface_TVD: order must be 0, 1 or 2.");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_over_interface_TVD, phi, q, 0, 0); CHKERRXX(ierr);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  double *mask_p;
  if (mask != NULL) {
    ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);
  } else {
    mask_p = phi_p;
  }

  if (band_use > 0) band_use = -band_use;

  int num_iters_min = 10;

  Vec b_qn_well_defined;  double *b_qn_well_defined_p;
  Vec b_qnn_well_defined; double *b_qnn_well_defined_p;

  double *q_p;

  Vec qx; double *qx_p;
  Vec qy; double *qy_p;
#ifdef P4_TO_P8
  Vec qz; double *qz_p;
#endif

  Vec qxx; double *qxx_p;
  Vec qyy; double *qyy_p;
  Vec qxy; double *qxy_p;
#ifdef P4_TO_P8
  Vec qzz; double *qzz_p;
  Vec qyz; double *qyz_p;
  Vec qzx; double *qzx_p;
#endif

  double dxyz[P4EST_DIM];

  dxyz_min(p4est, dxyz);

#ifdef P4_TO_P8
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]) + SQR(dxyz[2]));
#else
  double diag = sqrt(SQR(dxyz[0]) + SQR(dxyz[1]));
#endif

  double rel_thresh = 1.e-2;

  double tol_d  = tol/diag;
  double tol_dd = tol_d/diag;

  /* init the neighborhood information if needed */
  /* NOTE: from now on the neighbors will be initialized ... do we want to clear them
   * at the end of this function if they were not initialized beforehand ?
   */
  ngbd->init_neighbors();

  /* compute the normals */
  double *nx, *ny, *nz;

  if (normal != NULL)
  {
    ierr = VecGetArray(normal[0], &nx); CHKERRXX(ierr);
    ierr = VecGetArray(normal[1], &ny); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(normal[2], &nz); CHKERRXX(ierr);
#endif
  } else {
    nx = new double[nodes->num_owned_indeps];
    ny = new double[nodes->num_owned_indeps];
#ifdef P4_TO_P8
    nz = new double[nodes->num_owned_indeps];
#endif

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if (phi_p[n] > band_use && phi_p[n] < band_extend)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        nx[n] = qnnn.dx_central(phi_p);
        ny[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
        nz[n] = qnnn.dz_central(phi_p);
        double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n] + nz[n]*nz[n]);
#else
        double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n]);
#endif

        if(norm>EPS)
        {
          nx[n] /= norm;
          ny[n] /= norm;
#ifdef P4_TO_P8
          nz[n] /= norm;
#endif
        }
        else
        {
          nx[n] = 0;
          ny[n] = 0;
#ifdef P4_TO_P8
          nz[n] = 0;
#endif
        }
      }
    }
  }

  ierr = VecGetArray(q , &q_p) ; CHKERRXX(ierr);

  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  /* initialize pure derivatives */
  if(order >=1 )
  {
    // first-order derivatives
    if (q_d == NULL)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &qx);  CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &qy);  CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecCreateGhostNodes(p4est, nodes, &qz);  CHKERRXX(ierr);
#endif
    } else {
      qx = q_d[0];
      qy = q_d[1];
#ifdef P4_TO_P8
      qz = q_d[2];
#endif
    }

    ierr = VecGetArray(qx,  &qx_p);  CHKERRXX(ierr);
    ierr = VecGetArray(qy,  &qy_p);  CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qz,  &qz_p);  CHKERRXX(ierr);
#endif

    // second-order derivatives
    if (order == 2)
    {
      if (q_d == NULL)
      {
        ierr = VecCreateGhostNodes(p4est, nodes, &qxx); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
        ierr = VecCreateGhostNodes(p4est, nodes, &qzz); CHKERRXX(ierr);
#endif
      } else {
        qxx = q_dd[0];
        qyy = q_dd[1];
#ifdef P4_TO_P8
        qzz = q_dd[2];
#endif
      }

      ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif
    }

    ierr = VecCreateGhostNodes(p4est, nodes, &b_qn_well_defined); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for (int map_idx = 0; map_idx < 2; ++map_idx)
    {
      const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
      for(size_t n_map=0; n_map<map.size(); ++n_map)
      {
        p4est_locidx_t n = map[n_map];
        if (phi_p[n] > band_use)
        {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          if (qnnn.is_stencil_in_negative_domain(mask_p))
          {
            b_qn_well_defined_p[n] = -1;

            XCODE( qx_p[n] = qnnn.dx_central(q_p) );
            YCODE( qy_p[n] = qnnn.dy_central(q_p) );
            ZCODE( qz_p[n] = qnnn.dz_central(q_p) );

            if (order == 2)
            {
              XCODE( qxx_p[n] = qnnn.dxx_central(q_p) );
              YCODE( qyy_p[n] = qnnn.dyy_central(q_p) );
              ZCODE( qzz_p[n] = qnnn.dzz_central(q_p) );
            }
          }
          else if (mask_p[qnnn.node_000]<-EPS && bc != NULL)
          {
            b_qn_well_defined_p[n] = 1;

            if (use_nonzero_guess)
            {
              XCODE( qx_p[n] = qnnn.dx_central(q_p) );
              YCODE( qy_p[n] = qnnn.dy_central(q_p) );
              ZCODE( qz_p[n] = qnnn.dz_central(q_p) );
            } else if (q_d == NULL) {
              XCODE( qx_p[n] = 0 );
              YCODE( qy_p[n] = 0 );
              ZCODE( qz_p[n] = 0 );
            }

            if (order == 2)
            {
              if (use_nonzero_guess && phi_p[n] < band_check)
              {
                XCODE( qxx_p[n] = qnnn.dxx_central(q_p) );
                YCODE( qyy_p[n] = qnnn.dyy_central(q_p) );
                ZCODE( qzz_p[n] = qnnn.dzz_central(q_p) );
              } else if (q_dd == NULL || (use_nonzero_guess && phi_p[n] >= band_check)) {
                XCODE( qxx_p[n] = 0 );
                YCODE( qyy_p[n] = 0 );
                ZCODE( qzz_p[n] = 0 );
              }
            }

            // correct for boundary conditions if provided
            if (bc->type == DIRICHLET && bc->pointwise && bc->num_value_pts(n) < P4EST_DIM+1 && bc->num_value_pts(n) > 0)
            {
              double d_m00 = qnnn.d_m00, d_p00 = qnnn.d_p00;
              double d_0m0 = qnnn.d_0m0, d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
              double d_00m = qnnn.d_00m, d_00p = qnnn.d_00p;
#endif

              bool nei_m00 = mask_p[qnnn.neighbor_m00()] < -EPS, nei_p00 = mask_p[qnnn.neighbor_p00()] < -EPS;
              bool nei_0m0 = mask_p[qnnn.neighbor_0m0()] < -EPS, nei_0p0 = mask_p[qnnn.neighbor_0p0()] < -EPS;
#ifdef P4_TO_P8
              bool nei_00m = mask_p[qnnn.neighbor_00m()] < -EPS, nei_00p = mask_p[qnnn.neighbor_00p()] < -EPS;
#endif

              // assuming grid is uniform near the interface
              double q_m00 = qnnn.f_m00_linear(q_p), q_p00 = qnnn.f_p00_linear(q_p);
              double q_0m0 = qnnn.f_0m0_linear(q_p), q_0p0 = qnnn.f_0p0_linear(q_p);
#ifdef P4_TO_P8
              double q_00m = qnnn.f_00m_linear(q_p), q_00p = qnnn.f_00p_linear(q_p);
#endif
              double q_000 = q_p[n];

              double d_min = diag;
              for (unsigned int i = 0; i < bc->num_value_pts(n); ++i)
              {
                int idx = bc->idx_value_pt(n,i);
                interface_point_cartesian_t *pt = &bc->dirichlet_pts[idx];
                switch (pt->dir)
                {
                  case 0: d_m00 = pt->dist; q_m00 = bc->get_value_pw(n,i); nei_m00 = true; break;
                  case 1: d_p00 = pt->dist; q_p00 = bc->get_value_pw(n,i); nei_p00 = true; break;
                  case 2: d_0m0 = pt->dist; q_0m0 = bc->get_value_pw(n,i); nei_0m0 = true; break;
                  case 3: d_0p0 = pt->dist; q_0p0 = bc->get_value_pw(n,i); nei_0p0 = true; break;
#ifdef P4_TO_P8
                  case 4: d_00m = pt->dist; q_00m = bc->get_value_pw(n,i); nei_00m = true; break;
                  case 5: d_00p = pt->dist; q_00p = bc->get_value_pw(n,i); nei_00p = true; break;
#endif
                }

                d_min = MIN(d_min, pt->dist);
              }

              bool well_defined = nei_m00 && nei_p00 && nei_0m0 && nei_0p0 CODE3D( && nei_00m && nei_00p );

              if (d_min > rel_thresh*diag && well_defined)
              {
                b_qn_well_defined_p[n] = -1;
                XCODE( qx_p[n] = ((q_p00-q_000)*d_m00/d_p00 + (q_000-q_m00)*d_p00/d_m00)/(d_m00+d_p00) );
                YCODE( qy_p[n] = ((q_0p0-q_000)*d_0m0/d_0p0 + (q_000-q_0m0)*d_0p0/d_0m0)/(d_0m0+d_0p0) );
                ZCODE( qz_p[n] = ((q_00p-q_000)*d_00m/d_00p + (q_000-q_00m)*d_00p/d_00m)/(d_00m+d_00p) );
                if (order == 2)
                {
                  XCODE( qxx_p[n] = 2.*((q_p00-q_000)/d_p00 - (q_000-q_m00)/d_m00)/(d_m00+d_p00); );
                  YCODE( qyy_p[n] = 2.*((q_0p0-q_000)/d_0p0 - (q_000-q_0m0)/d_0m0)/(d_0m0+d_0p0); );
                  ZCODE( qzz_p[n] = 2.*((q_00p-q_000)/d_00p - (q_000-q_00m)/d_00m)/(d_00m+d_00p); );
                }
              }
            }
          }
          else
          {
            b_qn_well_defined_p[n] = 1;

            if (use_nonzero_guess)
            {
              XCODE( qx_p[n] = qnnn.dx_central(q_p) );
              YCODE( qy_p[n] = qnnn.dy_central(q_p) );
              ZCODE( qz_p[n] = qnnn.dz_central(q_p) );
            } else if (q_d == NULL) {
              XCODE( qx_p[n] = 0 );
              YCODE( qy_p[n] = 0 );
              ZCODE( qz_p[n] = 0 );
            }

            if (order == 2)
            {
              if (use_nonzero_guess && phi_p[n] < band_check)
              {
                XCODE( qxx_p[n] = qnnn.dxx_central(q_p) );
                YCODE( qyy_p[n] = qnnn.dyy_central(q_p) );
                ZCODE( qzz_p[n] = qnnn.dzz_central(q_p) );
              } else if (q_dd == NULL || (use_nonzero_guess && phi_p[n] >= band_check)) {
                XCODE( qxx_p[n] = 0 );
                YCODE( qyy_p[n] = 0 );
                ZCODE( qzz_p[n] = 0 );
              }
            }
          }
        }
        else
        {
          b_qn_well_defined_p[n] = -1;

          XCODE( qx_p[n] = 0 );
          YCODE( qy_p[n] = 0 );
          ZCODE( qz_p[n] = 0 );

          if (order == 2)
          {
            XCODE( qxx_p[n] = 0 );
            YCODE( qyy_p[n] = 0 );
            ZCODE( qzz_p[n] = 0 );
          }
        }
      }

      if (map_idx == 0)
      {
        /* initiate the communication */
        ierr = VecGhostUpdateBegin(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

        ierr = VecGhostUpdateBegin(qx,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(qy,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
        ierr = VecGhostUpdateBegin(qz,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

        if (order == 2)
        {
          ierr = VecGhostUpdateBegin(qxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(qyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
          ierr = VecGhostUpdateBegin(qzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
        }
      }
      else
      {
        /* end update communication */
        ierr = VecGhostUpdateEnd(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

        ierr = VecGhostUpdateEnd(qx,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(qy,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
        ierr = VecGhostUpdateEnd(qz,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
        if (order == 2)
        {
          ierr = VecGhostUpdateEnd(qxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(qyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
          ierr = VecGhostUpdateEnd(qzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif

        }
      }
    }

    ierr = VecRestoreArray(qx,  &qx_p);  CHKERRXX(ierr);
    ierr = VecRestoreArray(qy,  &qy_p);  CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qz,  &qz_p);  CHKERRXX(ierr);
#endif

    if (order == 2)
    {
      ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);

  /* initialize cross-derivatives */
  if(order == 2)
  {
    if (q_dd == NULL)
    {
      ierr = VecCreateGhostNodes(p4est, nodes, &qxy); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecCreateGhostNodes(p4est, nodes, &qyz); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &qzx); CHKERRXX(ierr);
#endif
    } else {
      qxy = q_dd[P4EST_DIM];
#ifdef P4_TO_P8
      qyz = q_dd[4];
      qzx = q_dd[5];
#endif
    }

    ierr = VecDuplicate(b_qn_well_defined, &b_qnn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qx, &qx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qy, &qy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qz, &qz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(qxy, &qxy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qyz, &qyz_p); CHKERRXX(ierr);
    ierr = VecGetArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    for (int map_idx = 0; map_idx < 2; ++map_idx)
    {
      const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
      for(size_t n_map=0; n_map<map.size(); ++n_map)
      {
        p4est_locidx_t n = map[n_map];
        if (phi_p[n] > band_use)
        {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          if (qnnn.is_stencil_in_negative_domain(b_qn_well_defined_p))
          {
            b_qnn_well_defined_p[n] = -1;

            qxy_p[n] = .5*(qnnn.dx_central(qy_p) + qnnn.dy_central(qx_p));
#ifdef P4_TO_P8
            qyz_p[n] = .5*(qnnn.dy_central(qz_p) + qnnn.dz_central(qy_p));
            qzx_p[n] = .5*(qnnn.dz_central(qx_p) + qnnn.dx_central(qz_p));
#endif
          } else {
            b_qnn_well_defined_p[n] = 1;

            if (use_nonzero_guess && phi_p[n] < band_check)
            {
              qxy_p[n] = .5*(qnnn.dx_central(qy_p) + qnnn.dy_central(qx_p));
#ifdef P4_TO_P8
              qyz_p[n] = .5*(qnnn.dy_central(qz_p) + qnnn.dz_central(qy_p));
              qzx_p[n] = .5*(qnnn.dz_central(qx_p) + qnnn.dx_central(qz_p));
#endif
            } else if (q_dd == NULL || (use_nonzero_guess && phi_p[n] >= band_check)) {
              qxy_p[n] = 0;
#ifdef P4_TO_P8
              qyz_p[n] = 0;
              qzx_p[n] = 0;
#endif
            }

          }
        }
        else
        {
          b_qnn_well_defined_p[n] = -1;
          qxy_p[n] = 0;
#ifdef P4_TO_P8
          qyz_p[n] = 0;
          qzx_p[n] = 0;
#endif
        }
      }

      if (map_idx == 0)
      {
        /* initiate the communication */
        ierr = VecGhostUpdateBegin(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(qxy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
        ierr = VecGhostUpdateBegin(qyz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(qzx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
      }
      else
      {
        /* end update communication */
        ierr = VecGhostUpdateEnd(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(qxy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
        ierr = VecGhostUpdateEnd(qyz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(qzx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
      }
    }

    ierr = VecRestoreArray(qx, &qx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qy, &qy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qz, &qz_p); CHKERRXX(ierr);
#endif

    ierr = VecRestoreArray(qxy, &qxy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qyz, &qyz_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif

    ierr = VecRestoreArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  /* extrapolate qnn */
  if(order==2)
  {
    ierr = VecGetArray(b_qn_well_defined,  &b_qn_well_defined_p);  CHKERRXX(ierr);
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    Vec tmp_xx; double *tmp_xx_p; ierr = VecDuplicate(qxx, &tmp_xx); CHKERRXX(ierr);
    Vec tmp_yy; double *tmp_yy_p; ierr = VecDuplicate(qyy, &tmp_yy); CHKERRXX(ierr);
    Vec tmp_xy; double *tmp_xy_p; ierr = VecDuplicate(qxy, &tmp_xy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    Vec tmp_zz; double *tmp_zz_p; ierr = VecDuplicate(qzz, &tmp_zz); CHKERRXX(ierr);
    Vec tmp_yz; double *tmp_yz_p; ierr = VecDuplicate(qyz, &tmp_yz); CHKERRXX(ierr);
    Vec tmp_zx; double *tmp_zx_p; ierr = VecDuplicate(qzx, &tmp_zx); CHKERRXX(ierr);
#endif

    VecCopyGhost(qxx, tmp_xx);
    VecCopyGhost(qyy, tmp_yy);
    VecCopyGhost(qxy, tmp_xy);
#ifdef P4_TO_P8
    VecCopyGhost(qzz, tmp_zz);
    VecCopyGhost(qyz, tmp_yz);
    VecCopyGhost(qzx, tmp_zx);
#endif

    int    it     = 0;
//    double change = 1;
//    double initial_change = 0;
//    while (it < iterations && change > 1.e-4*initial_change || it < num_iters_min)
    double change = tol_dd+1;
    while (it < iterations && change > tol_dd || it < num_iters_min)
    {
      change = 0;
      ++it;
      ierr = VecGetArray(tmp_xx, &tmp_xx_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_yy, &tmp_yy_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_xy, &tmp_xy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(tmp_zz, &tmp_zz_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_yz, &tmp_yz_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_zx, &tmp_zx_p); CHKERRXX(ierr);
#endif

      ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
      ierr = VecGetArray(qxy, &qxy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
      ierr = VecGetArray(qyz, &qyz_p); CHKERRXX(ierr);
      ierr = VecGetArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif

      for (int map_idx = 0; map_idx < 2; ++map_idx)
      {
        const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
        for(size_t n_map=0; n_map<map.size(); ++n_map)
        {
          p4est_locidx_t n = map[n_map];
          // pure derivatives
          if (b_qn_well_defined_p[n] > 0 && phi_p[n] < band_extend)
          {
            const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
            double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
            dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
            dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
            dt /= 3;
#else
            dt /= 2;
#endif

            /* first order one sided derivative */
            double qxxd_dot_n = 0;
            double qyyd_dot_n = 0;
#ifdef P4_TO_P8
            double qzzd_dot_n = 0;
#endif

            qxxd_dot_n += nx[n]*(nx[n]>0 ? (qxx_p[n] - qnnn.inl_f_m00_linear(qxx_p)) / qnnn.d_m00 : (qnnn.inl_f_p00_linear(qxx_p) - qxx_p[n]) / qnnn.d_p00);
            qxxd_dot_n += ny[n]*(ny[n]>0 ? (qxx_p[n] - qnnn.inl_f_0m0_linear(qxx_p)) / qnnn.d_0m0 : (qnnn.inl_f_0p0_linear(qxx_p) - qxx_p[n]) / qnnn.d_0p0);
#ifdef P4_TO_P8
            qxxd_dot_n += nz[n]*(nz[n]>0 ? (qxx_p[n] - qnnn.inl_f_00m_linear(qxx_p)) / qnnn.d_00m : (qnnn.inl_f_00p_linear(qxx_p) - qxx_p[n]) / qnnn.d_00p);
#endif

            qyyd_dot_n += nx[n]*(nx[n]>0 ? (qyy_p[n] - qnnn.inl_f_m00_linear(qyy_p)) / qnnn.d_m00 : (qnnn.inl_f_p00_linear(qyy_p) - qyy_p[n]) / qnnn.d_p00);
            qyyd_dot_n += ny[n]*(ny[n]>0 ? (qyy_p[n] - qnnn.inl_f_0m0_linear(qyy_p)) / qnnn.d_0m0 : (qnnn.inl_f_0p0_linear(qyy_p) - qyy_p[n]) / qnnn.d_0p0);
#ifdef P4_TO_P8
            qyyd_dot_n += nz[n]*(nz[n]>0 ? (qyy_p[n] - qnnn.inl_f_00m_linear(qyy_p)) / qnnn.d_00m : (qnnn.inl_f_00p_linear(qyy_p) - qyy_p[n]) / qnnn.d_00p);

            qzzd_dot_n += nx[n]*(nx[n]>0 ? (qzz_p[n] - qnnn.inl_f_m00_linear(qzz_p)) / qnnn.d_m00 : (qnnn.inl_f_p00_linear(qzz_p) - qzz_p[n]) / qnnn.d_p00);
            qzzd_dot_n += ny[n]*(ny[n]>0 ? (qzz_p[n] - qnnn.inl_f_0m0_linear(qzz_p)) / qnnn.d_0m0 : (qnnn.inl_f_0p0_linear(qzz_p) - qzz_p[n]) / qnnn.d_0p0);
            qzzd_dot_n += nz[n]*(nz[n]>0 ? (qzz_p[n] - qnnn.inl_f_00m_linear(qzz_p)) / qnnn.d_00m : (qnnn.inl_f_00p_linear(qzz_p) - qzz_p[n]) / qnnn.d_00p);
#endif
            qxxd_dot_n *= dt;
            qyyd_dot_n *= dt;
#ifdef P4_TO_P8
            qzzd_dot_n *= dt;
#endif

            tmp_xx_p[n] = qxx_p[n] - qxxd_dot_n;
            tmp_yy_p[n] = qyy_p[n] - qyyd_dot_n;
#ifdef P4_TO_P8
            tmp_zz_p[n] = qzz_p[n] - qzzd_dot_n;
#endif
            if (phi_p[n] < band_check)
            {
              if (fabs(qxxd_dot_n) > change) change = fabs(qxxd_dot_n);
              if (fabs(qyyd_dot_n) > change) change = fabs(qyyd_dot_n);
#ifdef P4_TO_P8
              if (fabs(qzzd_dot_n) > change) change = fabs(qzzd_dot_n);
#endif
            }
          }

          // cross derivatives
          if (b_qnn_well_defined_p[n] > 0 && phi_p[n] < band_extend)
          {
            const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
            double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
            dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
            dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
            dt /= 3;
#else
            dt /= 2;
#endif

            /* first order one sided derivative */
            double qxyd_dot_n = 0;
#ifdef P4_TO_P8
            double qyzd_dot_n = 0;
            double qzxd_dot_n = 0;
#endif

            qxyd_dot_n += nx[n]*(nx[n]>0 ? (qxy_p[n] - qnnn.inl_f_m00_linear(qxy_p)) / qnnn.d_m00 : (qnnn.inl_f_p00_linear(qxy_p) - qxy_p[n]) / qnnn.d_p00);
            qxyd_dot_n += ny[n]*(ny[n]>0 ? (qxy_p[n] - qnnn.inl_f_0m0_linear(qxy_p)) / qnnn.d_0m0 : (qnnn.inl_f_0p0_linear(qxy_p) - qxy_p[n]) / qnnn.d_0p0);
#ifdef P4_TO_P8
            qxyd_dot_n += nz[n]*(nz[n]>0 ? (qxy_p[n] - qnnn.inl_f_00m_linear(qxy_p)) / qnnn.d_00m : (qnnn.inl_f_00p_linear(qxy_p) - qxy_p[n]) / qnnn.d_00p);

            qyzd_dot_n += nx[n]*(nx[n]>0 ? (qyz_p[n] - qnnn.inl_f_m00_linear(qyz_p)) / qnnn.d_m00 : (qnnn.inl_f_p00_linear(qyz_p) - qyz_p[n]) / qnnn.d_p00);
            qyzd_dot_n += ny[n]*(ny[n]>0 ? (qyz_p[n] - qnnn.inl_f_0m0_linear(qyz_p)) / qnnn.d_0m0 : (qnnn.inl_f_0p0_linear(qyz_p) - qyz_p[n]) / qnnn.d_0p0);
            qyzd_dot_n += nz[n]*(nz[n]>0 ? (qyz_p[n] - qnnn.inl_f_00m_linear(qyz_p)) / qnnn.d_00m : (qnnn.inl_f_00p_linear(qyz_p) - qyz_p[n]) / qnnn.d_00p);

            qzxd_dot_n += nx[n]*(nx[n]>0 ? (qzx_p[n] - qnnn.inl_f_m00_linear(qzx_p)) / qnnn.d_m00 : (qnnn.inl_f_p00_linear(qzx_p) - qzx_p[n]) / qnnn.d_p00);
            qzxd_dot_n += ny[n]*(ny[n]>0 ? (qzx_p[n] - qnnn.inl_f_0m0_linear(qzx_p)) / qnnn.d_0m0 : (qnnn.inl_f_0p0_linear(qzx_p) - qzx_p[n]) / qnnn.d_0p0);
            qzxd_dot_n += nz[n]*(nz[n]>0 ? (qzx_p[n] - qnnn.inl_f_00m_linear(qzx_p)) / qnnn.d_00m : (qnnn.inl_f_00p_linear(qzx_p) - qzx_p[n]) / qnnn.d_00p);
#endif

            qxyd_dot_n *= dt;
#ifdef P4_TO_P8
            qyzd_dot_n *= dt;
            qzxd_dot_n *= dt;
#endif

            tmp_xy_p[n] = qxy_p[n] - qxyd_dot_n;
#ifdef P4_TO_P8
            tmp_yz_p[n] = qyz_p[n] - qyzd_dot_n;
            tmp_zx_p[n] = qzx_p[n] - qzxd_dot_n;
#endif
            if (phi_p[n] < band_check)
            {
              if (fabs(qxyd_dot_n) > change) change = fabs(qxyd_dot_n);
#ifdef P4_TO_P8
              if (fabs(qyzd_dot_n) > change) change = fabs(qyzd_dot_n);
              if (fabs(qzxd_dot_n) > change) change = fabs(qzxd_dot_n);
#endif
            }
          }
        }

        if (map_idx == 0)
        {
          /* initiate the communication */
          ierr = VecGhostUpdateBegin(tmp_xx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(tmp_yy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(tmp_xy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #ifdef P4_TO_P8
          ierr = VecGhostUpdateBegin(tmp_zz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(tmp_yz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(tmp_zx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #endif
        } else {
          /* get global max change */
          int mpiret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

          /* end update communication */
          ierr = VecGhostUpdateEnd(tmp_xx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(tmp_yy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(tmp_xy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #ifdef P4_TO_P8
          ierr = VecGhostUpdateEnd(tmp_zz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(tmp_yz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(tmp_zx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #endif
        }
      }

//      if (it == 1) initial_change = change;

      ierr = VecRestoreArray(tmp_xx, &tmp_xx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_yy, &tmp_yy_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_xy, &tmp_xy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(tmp_zz, &tmp_zz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_yz, &tmp_yz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_zx, &tmp_zx_p); CHKERRXX(ierr);
#endif

      ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qxy, &qxy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qyz, &qyz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif

      if (verbose)
      {
        ierr = PetscPrintf(p4est->mpicomm, "Extending second derivative, iteration: %d, error: %e\n", it, change); CHKERRXX(ierr);
      }

      Vec swap_tmp;
      swap_tmp = tmp_xx; tmp_xx = qxx; qxx = swap_tmp;
      swap_tmp = tmp_yy; tmp_yy = qyy; qyy = swap_tmp;
      swap_tmp = tmp_xy; tmp_xy = qxy; qxy = swap_tmp;
#ifdef P4_TO_P8
      swap_tmp = tmp_zz; tmp_zz = qzz; qzz = swap_tmp;
      swap_tmp = tmp_yz; tmp_yz = qyz; qyz = swap_tmp;
      swap_tmp = tmp_zx; tmp_zx = qzx; qzx = swap_tmp;
#endif
    }

    if (it%2 != 0)
    {
      Vec swap_tmp;
      swap_tmp = tmp_xx; tmp_xx = qxx; qxx = swap_tmp;
      swap_tmp = tmp_yy; tmp_yy = qyy; qyy = swap_tmp;
      swap_tmp = tmp_xy; tmp_xy = qxy; qxy = swap_tmp;
#ifdef P4_TO_P8
      swap_tmp = tmp_zz; tmp_zz = qzz; qzz = swap_tmp;
      swap_tmp = tmp_yz; tmp_yz = qyz; qyz = swap_tmp;
      swap_tmp = tmp_zx; tmp_zx = qzx; qzx = swap_tmp;
#endif

      copy_ghosted_vec(tmp_xx, qxx);
      copy_ghosted_vec(tmp_yy, qyy);
      copy_ghosted_vec(tmp_xy, qxy);
#ifdef P4_TO_P8
      copy_ghosted_vec(tmp_zz, qzz);
      copy_ghosted_vec(tmp_yz, qyz);
      copy_ghosted_vec(tmp_zx, qzx);
#endif
    }

    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qn_well_defined,  &b_qn_well_defined_p);  CHKERRXX(ierr);

    ierr = VecDestroy(tmp_xx); CHKERRXX(ierr);
    ierr = VecDestroy(tmp_yy); CHKERRXX(ierr);
    ierr = VecDestroy(tmp_xy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(tmp_zz); CHKERRXX(ierr);
    ierr = VecDestroy(tmp_yz); CHKERRXX(ierr);
    ierr = VecDestroy(tmp_zx); CHKERRXX(ierr);
#endif
  }

  /* extrapolate qn */
  if (order >= 1)
  {
    if (order == 2)
    {
      ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
      ierr = VecGetArray(qxy, &qxy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
      ierr = VecGetArray(qyz, &qyz_p); CHKERRXX(ierr);
      ierr = VecGetArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif
    }

    Vec tmp_x; double *tmp_x_p; ierr = VecDuplicate(qx, &tmp_x); CHKERRXX(ierr);
    Vec tmp_y; double *tmp_y_p; ierr = VecDuplicate(qy, &tmp_y); CHKERRXX(ierr);
#ifdef P4_TO_P8
    Vec tmp_z; double *tmp_z_p; ierr = VecDuplicate(qz, &tmp_z); CHKERRXX(ierr);
#endif

    VecCopyGhost(qx, tmp_x);
    VecCopyGhost(qy, tmp_y);
#ifdef P4_TO_P8
    VecCopyGhost(qz, tmp_z);
#endif

    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    int    it     = 0;
    double change = tol+1;
    while (it < iterations && change > tol_d || it < num_iters_min)
    {
      change = 0;
      ++it;
      ierr = VecGetArray(qx, &qx_p); CHKERRXX(ierr);
      ierr = VecGetArray(qy, &qy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(qz, &qz_p); CHKERRXX(ierr);
#endif

      ierr = VecGetArray(tmp_x, &tmp_x_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_y, &tmp_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(tmp_z, &tmp_z_p); CHKERRXX(ierr);
#endif

      for (int map_idx = 0; map_idx < 2; ++map_idx)
      {
        const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
        for(size_t n_map=0; n_map<map.size(); ++n_map)
        {
          p4est_locidx_t n = map[n_map];
          if(b_qn_well_defined_p[n] > 0 && phi_p[n] < band_extend)
          {
            const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
            double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
            dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
            dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
            dt /= 3;
#else
            dt /= 2;
#endif

            /* first order one sided derivative */
            double qxd_dot_n = 0;
            double qyd_dot_n = 0;
#ifdef P4_TO_P8
            double qzd_dot_n = 0;
#endif

            qxd_dot_n += nx[n]*((nx[n]>0 ? (qx_p[n] - qnnn.inl_f_m00_linear(qx_p)) / qnnn.d_m00 : (qnnn.inl_f_p00_linear(qx_p) - qx_p[n]) / qnnn.d_p00) - (order == 2 ? qxx_p[n] : 0));
            qxd_dot_n += ny[n]*((ny[n]>0 ? (qx_p[n] - qnnn.inl_f_0m0_linear(qx_p)) / qnnn.d_0m0 : (qnnn.inl_f_0p0_linear(qx_p) - qx_p[n]) / qnnn.d_0p0) - (order == 2 ? qxy_p[n] : 0));
#ifdef P4_TO_P8
            qxd_dot_n += nz[n]*((nz[n]>0 ? (qx_p[n] - qnnn.inl_f_00m_linear(qx_p)) / qnnn.d_00m : (qnnn.inl_f_00p_linear(qx_p) - qx_p[n]) / qnnn.d_00p) - (order == 2 ? qzx_p[n] : 0));
#endif

            qyd_dot_n += nx[n]*((nx[n]>0 ? (qy_p[n] - qnnn.inl_f_m00_linear(qy_p)) / qnnn.d_m00 : (qnnn.inl_f_p00_linear(qy_p) - qy_p[n]) / qnnn.d_p00) - (order == 2 ? qxy_p[n] : 0));
            qyd_dot_n += ny[n]*((ny[n]>0 ? (qy_p[n] - qnnn.inl_f_0m0_linear(qy_p)) / qnnn.d_0m0 : (qnnn.inl_f_0p0_linear(qy_p) - qy_p[n]) / qnnn.d_0p0) - (order == 2 ? qyy_p[n] : 0));
#ifdef P4_TO_P8
            qyd_dot_n += nz[n]*((nz[n]>0 ? (qy_p[n] - qnnn.inl_f_00m_linear(qy_p)) / qnnn.d_00m : (qnnn.inl_f_00p_linear(qy_p) - qy_p[n]) / qnnn.d_00p) - (order == 2 ? qyz_p[n] : 0));

            qzd_dot_n += nx[n]*((nx[n]>0 ? (qz_p[n] - qnnn.inl_f_m00_linear(qz_p)) / qnnn.d_m00 : (qnnn.inl_f_p00_linear(qz_p) - qz_p[n]) / qnnn.d_p00) - (order == 2 ? qzx_p[n] : 0));
            qzd_dot_n += ny[n]*((ny[n]>0 ? (qz_p[n] - qnnn.inl_f_0m0_linear(qz_p)) / qnnn.d_0m0 : (qnnn.inl_f_0p0_linear(qz_p) - qz_p[n]) / qnnn.d_0p0) - (order == 2 ? qyz_p[n] : 0));
            qzd_dot_n += nz[n]*((nz[n]>0 ? (qz_p[n] - qnnn.inl_f_00m_linear(qz_p)) / qnnn.d_00m : (qnnn.inl_f_00p_linear(qz_p) - qz_p[n]) / qnnn.d_00p) - (order == 2 ? qzz_p[n] : 0));
#endif

            qxd_dot_n *= dt;
            qyd_dot_n *= dt;
#ifdef P4_TO_P8
            qzd_dot_n *= dt;
#endif
            tmp_x_p[n] = qx_p[n] - qxd_dot_n;
            tmp_y_p[n] = qy_p[n] - qyd_dot_n;
#ifdef P4_TO_P8
            tmp_z_p[n] = qz_p[n] - qzd_dot_n;
#endif
            if (phi_p[n] < band_check)
            {
              if (fabs(qxd_dot_n) > change) change = fabs(qxd_dot_n);
              if (fabs(qyd_dot_n) > change) change = fabs(qyd_dot_n);
#ifdef P4_TO_P8
              if (fabs(qzd_dot_n) > change) change = fabs(qzd_dot_n);
#endif
            }
          }
        }

        if (map_idx == 0)
        {
          /* initiate the communication */
          ierr = VecGhostUpdateBegin(tmp_x, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(tmp_y, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #ifdef P4_TO_P8
          ierr = VecGhostUpdateBegin(tmp_z, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #endif
        } else {
          /* get global max change */
          int mpiret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

          /* end update communication */
          ierr = VecGhostUpdateEnd(tmp_x, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(tmp_y, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #ifdef P4_TO_P8
          ierr = VecGhostUpdateEnd(tmp_z, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #endif
        }
      }

      ierr = VecRestoreArray(qx, &qx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qy, &qy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(qz, &qz_p); CHKERRXX(ierr);
#endif

      ierr = VecRestoreArray(tmp_x, &tmp_x_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_y, &tmp_y_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(tmp_z, &tmp_z_p); CHKERRXX(ierr);
#endif

      if (verbose)
      {
        ierr = PetscPrintf(p4est->mpicomm, "Extending first derivative, iteration: %d, error: %e\n", it, change); CHKERRXX(ierr);
      }

      Vec swap_tmp;
      swap_tmp = tmp_x; tmp_x = qx; qx = swap_tmp;
      swap_tmp = tmp_y; tmp_y = qy; qy = swap_tmp;
#ifdef P4_TO_P8
      swap_tmp = tmp_z; tmp_z = qz; qz = swap_tmp;
#endif
    }

    if (it%2 != 0)
    {
      Vec swap_tmp;
      swap_tmp = tmp_x; tmp_x = qx; qx = swap_tmp;
      swap_tmp = tmp_y; tmp_y = qy; qy = swap_tmp;
#ifdef P4_TO_P8
      swap_tmp = tmp_z; tmp_z = qz; qz = swap_tmp;
#endif

      copy_ghosted_vec(tmp_x, qx);
      copy_ghosted_vec(tmp_y, qy);
#ifdef P4_TO_P8
      copy_ghosted_vec(tmp_z, qz);
#endif
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    if (order == 2)
    {
      ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qxy, &qxy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qyz, &qyz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif
    }

    ierr = VecDestroy(tmp_x); CHKERRXX(ierr);
    ierr = VecDestroy(tmp_y); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(tmp_z); CHKERRXX(ierr);
#endif
  }

  if (order >= 1) { ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr); }
  if (order == 2) { ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr); }

  /* extrapolate q */
//  Vec Qxx; double *Qxx_p; ierr = VecCreateGhostNodes(p4est, nodes, &Qxx); CHKERRXX(ierr);
//  Vec Qyy; double *Qyy_p; ierr = VecCreateGhostNodes(p4est, nodes, &Qyy); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//  Vec Qzz; double *Qzz_p; ierr = VecCreateGhostNodes(p4est, nodes, &Qzz); CHKERRXX(ierr);
//#endif

//  VecSetGhost(Qxx, 0.);
//  VecSetGhost(Qyy, 0.);
//#ifdef P4_TO_P8
//  VecSetGhost(Qzz, 0.);
//#endif

  if (order >= 1)
  {
    ierr = VecGetArray(qx, &qx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qy, &qy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qz, &qz_p); CHKERRXX(ierr);
#endif
  }

  Vec tmp; double *tmp_p; ierr = VecDuplicate(q, &tmp); CHKERRXX(ierr);
  VecCopyGhost(q, tmp);

  int    it     = 0;
  double change = tol+1;
  while (it < iterations && change > tol || it < num_iters_min)
  {
    change = 0;
    ++it;
    if (order != 2)
    {
//#ifdef P4_TO_P8
//      ngbd->second_derivatives_central(q, Qxx, Qyy, Qzz);
//#else
//      ngbd->second_derivatives_central(q, Qxx, Qyy);
//#endif
//      ierr = VecGetArray(Qxx, &qxx_p); CHKERRXX(ierr);
//      ierr = VecGetArray(Qyy, &qyy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//      ierr = VecGetArray(Qzz, &qzz_p); CHKERRXX(ierr);
//#endif
    } else {
      ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif
    }

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    for (int map_idx = 0; map_idx < 2; ++map_idx)
    {
      const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
      for(size_t n_map=0; n_map<map.size(); ++n_map)
      {
        p4est_locidx_t n = map[n_map];
        if(mask_p[n] > -EPS && phi_p[n] < band_extend)
        {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivatives */
          double Qx = nx[n]>0 ? (q_p[n] - qnnn.inl_f_m00_linear(q_p)) / qnnn.d_m00
                              :-(q_p[n] - qnnn.inl_f_p00_linear(q_p)) / qnnn.d_p00;
          double Qy = ny[n]>0 ? (q_p[n] - qnnn.inl_f_0m0_linear(q_p)) / qnnn.d_0m0
                              :-(q_p[n] - qnnn.inl_f_0p0_linear(q_p)) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double Qz = nz[n]>0 ? (q_p[n] - qnnn.inl_f_00m_linear(q_p)) / qnnn.d_00m
                              :-(q_p[n] - qnnn.inl_f_00p_linear(q_p)) / qnnn.d_00p;
#endif

          /* second order derivatives */
          if (order == 2)
          {
            double qxx_m00 = qnnn.inl_f_m00_linear(qxx_p);
            double qxx_p00 = qnnn.inl_f_p00_linear(qxx_p);
            double qyy_0m0 = qnnn.inl_f_0m0_linear(qyy_p);
            double qyy_0p0 = qnnn.inl_f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
            double qzz_00m = qnnn.inl_f_00m_linear(qzz_p);
            double qzz_00p = qnnn.inl_f_00p_linear(qzz_p);
#endif

            /* minmod operation */
            qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
            qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
            qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
            qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
            qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
            qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

            if(nx[n]<0) Qx -= .5*qnnn.d_p00*qxx_p00;
            else        Qx += .5*qnnn.d_m00*qxx_m00;
            if(ny[n]<0) Qy -= .5*qnnn.d_0p0*qyy_0p0;
            else        Qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
            if(nz[n]<0) Qz -= .5*qnnn.d_00p*qzz_00p;
            else        Qz += .5*qnnn.d_00m*qzz_00m;
#endif
          }
          double change_loc = dt* SUMD( nx[n]*Qx, ny[n]*Qy,nz[n]*Qz )
                              - (order >= 1 ? dt* SUMD(nx[n]*qx_p[n], ny[n]*qy_p[n], nz[n]*qz_p[n]) : 0);
          if (phi_p[n] < band_check)
          {
            if (fabs(change_loc) > change) change = fabs(change_loc);
          }
          tmp_p[n] = q_p[n] - change_loc;
        }
      }

      if (map_idx == 0) {
        /* initiate the communication */
        ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      } else {
        /* get global max change */
        int mpiret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

        /* end update communication */
        ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
    }

    if (order != 2)
    {
//      ierr = VecRestoreArray(Qxx, &qxx_p); CHKERRXX(ierr);
//      ierr = VecRestoreArray(Qyy, &qyy_p); CHKERRXX(ierr);
//  #ifdef P4_TO_P8
//      ierr = VecRestoreArray(Qzz, &qzz_p); CHKERRXX(ierr);
//  #endif
    } else {
      ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
  #ifdef P4_TO_P8
      ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
  #endif
    }

    ierr = VecRestoreArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

    if (verbose)
    {
      ierr = PetscPrintf(p4est->mpicomm, "Extending values, iteration: %d, error: %e\n", it, change); CHKERRXX(ierr);
    }

    Vec swap_tmp = tmp; tmp = q; q = swap_tmp;
  }

  if (it%2 != 0)
  {
    Vec swap_tmp = tmp; tmp = q; q = swap_tmp;
    copy_ghosted_vec(tmp, q);
  }

  ierr = VecDestroy(tmp); CHKERRXX(ierr);

  if(order>=1)
  {
    ierr = VecRestoreArray(qx, &qx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qy, &qy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qz, &qz_p); CHKERRXX(ierr);
#endif

    if (q_d == NULL)
    {
      ierr = VecDestroy(qx); CHKERRXX(ierr);
      ierr = VecDestroy(qy); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecDestroy(qz); CHKERRXX(ierr);
#endif
    }
  }

  if (order == 2 && q_dd == NULL)
  {
    ierr = VecDestroy(qxx); CHKERRXX(ierr);
    ierr = VecDestroy(qyy); CHKERRXX(ierr);
    ierr = VecDestroy(qxy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(qzz); CHKERRXX(ierr);
    ierr = VecDestroy(qyz); CHKERRXX(ierr);
    ierr = VecDestroy(qzx); CHKERRXX(ierr);
#endif
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  if (mask != NULL) {
    ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
  }

//  ierr = VecDestroy(Qxx); CHKERRXX(ierr);
//  ierr = VecDestroy(Qyy); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//  ierr = VecDestroy(Qzz); CHKERRXX(ierr);
//#endif

  if (normal != NULL)
  {
    ierr = VecRestoreArray(normal[0], &nx); CHKERRXX(ierr);
    ierr = VecRestoreArray(normal[1], &ny); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(normal[2], &nz); CHKERRXX(ierr);
#endif
  } else {
    delete[] nx;
    delete[] ny;
#ifdef P4_TO_P8
    delete[] nz;
#endif
  }

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_over_interface_TVD, phi, q, 0, 0); CHKERRXX(ierr);
}


void my_p4est_level_set_t::extend_Over_Interface_TVD_not_parallel(Vec phi, Vec q, int iterations, int order) const
{
  PetscErrorCode ierr;

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  Vec qn, qnn;
  double *q_p, *qn_p, *qnn_p;
  Vec b_qn_well_defined;
  Vec b_qnn_well_defined;
  double *b_qn_well_defined_p;
  double *b_qnn_well_defined_p;

  /* compute the normals */
  std::vector<double> nx(nodes->num_owned_indeps);
  std::vector<double> ny(nodes->num_owned_indeps);
#ifdef P4_TO_P8
  std::vector<double> nz(nodes->num_owned_indeps);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);
    nx[n] = qnnn.dx_central(phi_p);
    ny[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    nz[n] = qnnn.dz_central(phi_p);
    double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n] + nz[n]*nz[n]);
#else
    double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n]);
#endif

    if(norm>EPS)
    {
      nx[n] /= norm;
      ny[n] /= norm;
#ifdef P4_TO_P8
      nz[n] /= norm;
#endif
    }
    else
    {
      nx[n] = 0;
      ny[n] = 0;
#ifdef P4_TO_P8
      nz[n] = 0;
#endif
    }
  }

  ierr = VecGetArray(q , &q_p) ; CHKERRXX(ierr);

  /* initialize qn */
  if(order >=1 )
  {
    ierr = VecDuplicate(phi, &qn); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &b_qn_well_defined ); CHKERRXX(ierr);

    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      ngbd->get_neighbors(n, qnnn);
      if(  phi_p[qnnn.node_000]<-EPS &&
     #ifdef P4_TO_P8
           ( phi_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( phi_p[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( phi_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( phi_p[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( phi_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( phi_p[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( phi_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( phi_p[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( phi_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( phi_p[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( phi_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( phi_p[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( phi_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( phi_p[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( phi_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( phi_p[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( phi_p[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( phi_p[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( phi_p[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( phi_p[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( phi_p[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( phi_p[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( phi_p[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( phi_p[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( phi_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
           ( phi_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
           ( phi_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
           ( phi_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
           ( phi_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( phi_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( phi_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( phi_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        qn_p[n] = ( nx[n]*qnnn.dx_central(q_p) +
                    ny[n]*qnnn.dy_central(q_p)
            #ifdef P4_TO_P8
                    + nz[n]*qnnn.dz_central(q_p)
            #endif
                    );
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0;
      }
    }

    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }


  /* initialize qnn */
  if(order == 2)
  {
    ierr = VecDuplicate(phi, &qnn ); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &b_qnn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecGetArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000]==true &&
     #ifdef P4_TO_P8
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_mp]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pp]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mp]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pp]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mp]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pp]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mp]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pp]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00m_mm]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_mp]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pm]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pp]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00p_mm]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_mp]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pm]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pp]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        qnn_p[n] = ( nx[n]*qnnn.dx_central(qn_p) +
                     ny[n]*qnnn.dy_central(qn_p)
             #ifdef P4_TO_P8
                     + nz[n]*qnnn.dz_central(qn_p)
             #endif
                     );
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0;
      }
    }

    ierr = VecRestoreArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  /* extrapolate qnn */
  if(order==2)
  {
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
    for(int it=0; it<iterations; ++it)
    {
      ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);

      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnnx = nx[n]>0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n]>0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n]>0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif

          qnn_p[n] -= ( dt*nx[n]*qnnx +
                        dt*ny[n]*qnny
              #ifdef P4_TO_P8
                        + dt*nz[n]*qnnz
              #endif
                        );
        }
      }
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  /* extrapolate qn */
  if(order>=1)
  {
    if(order==2) ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for(int it=0; it<iterations; ++it)
    {
      ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);

      for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      {
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnx = nx[n]>0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                               : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n]>0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                               : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n]>0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                               : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif

          qn_p[n] -= ( dt*nx[n]*qnx +
                       dt*ny[n]*qny
             #ifdef P4_TO_P8
                       + dt*nz[n]*qnz
             #endif
                       - (order==2 ? dt*qnn_p[n] : 0) );
        }
      }
      ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
    if(order==2) ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);

  if(order>=1) ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr);
  if(order==2) ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr);

  /* extrapolate q */
  Vec qxx, qyy;
  double *qxx_p, *qyy_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qxx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec qzz;
  double *qzz_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qzz); CHKERRXX(ierr);
#endif

  if(order>=1) ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);

  for(int it=0; it<iterations; ++it)
  {
#ifdef P4_TO_P8
    ngbd->second_derivatives_central(q, qxx, qyy, qzz);
#else
    ngbd->second_derivatives_central(q, qxx, qyy);
#endif

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      if(phi_p[n] > -EPS)
      {
        ngbd->get_neighbors(n, qnnn);
        double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
        dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
        dt /= 3;
#else
        dt /= 2;
#endif

        /* first order one sided derivatives */
        double qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                            : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                            : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                            : (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        /* second order derivatives */
        double qxx_m00 = qnnn.f_m00_linear(qxx_p);
        double qxx_p00 = qnnn.f_p00_linear(qxx_p);
        double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
        double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
        double qzz_00m = qnnn.f_00m_linear(qzz_p);
        double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

        /* minmod operation */
        qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
        else        qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
        else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

        q_p[n] -= ( dt*nx[n]*qx +
                    dt*ny[n]*qy
            #ifdef P4_TO_P8
                    + dt*nz[n]*qz
            #endif
                    - (order>=1 ? dt*qn_p[n] : 0) );
      }
    }

    ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(q  , &q_p  ); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(q, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (q, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  if (order >= 1) {
    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecDestroy(qn); CHKERRXX(ierr);
    ierr = VecDestroy(b_qn_well_defined); CHKERRXX(ierr);
  }
  if (order == 2) {
    ierr = VecDestroy(qnn); CHKERRXX(ierr);
    ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecDestroy(qxx); CHKERRXX(ierr);
  ierr = VecDestroy(qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(qzz); CHKERRXX(ierr);
#endif
}


void my_p4est_level_set_t::extend_from_interface_to_whole_domain_TVD_one_iteration( const std::vector<int>& map, double *phi_p,
                                                                                    std::vector<double>& nx, std::vector<double>& ny,
                                                                                    #ifdef P4_TO_P8
                                                                                    std::vector<double>& nz,
                                                                                    #endif
                                                                                    double *q_out_p,
                                                                                    double *q_p, double *qxx_p, double *qyy_p,
                                                                                    #ifdef P4_TO_P8
                                                                                    double *qzz_p,
                                                                                    #endif
                                                                                    std::vector<double>& qi_m00, std::vector<double>& qi_p00,
                                                                                    std::vector<double>& qi_0m0, std::vector<double>& qi_0p0,
                                                                                    #ifdef P4_TO_P8
                                                                                    std::vector<double>& qi_00m, std::vector<double>& qi_00p,
                                                                                    #endif
                                                                                    std::vector<double>& s_m00 , std::vector<double>& s_p00,
                                                                                    std::vector<double>& s_0m0 , std::vector<double>& s_0p0
                                                                                    #ifdef P4_TO_P8
                                                                                    , std::vector<double>& s_00m, std::vector<double>& s_00p
                                                                                    #endif
                                                                                    ) const
{
  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t n_map=0; n_map<map.size(); ++n_map)
  {
    p4est_locidx_t n = map[n_map];
    ngbd->get_neighbors(n, qnnn);
    //---------------------------------------------------------------------
    // Neighborhood information
    //---------------------------------------------------------------------
    double p_000, p_m00, p_p00, p_0m0, p_0p0;
    double q_000, q_m00, q_p00, q_0m0, q_0p0;
#ifdef P4_TO_P8
    double p_00m, p_00p;
    double q_00m, q_00p;
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p);
    qnnn.ngbd_with_quadratic_interpolation(q_p  , q_000, q_m00, q_p00, q_0m0, q_0p0, q_00m, q_00p);
#else
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0);
    qnnn.ngbd_with_quadratic_interpolation(q_p  , q_000, q_m00, q_p00, q_0m0, q_0p0);
#endif

    double s_p00_ = qnnn.d_p00; double s_m00_ = qnnn.d_m00;
    double s_0p0_ = qnnn.d_0p0; double s_0m0_ = qnnn.d_0m0;
#ifdef P4_TO_P8
    double s_00p_ = qnnn.d_00p; double s_00m_ = qnnn.d_00m;
#endif

    if(p_000*p_m00<0) {
      s_m00_ = s_m00[n];
      q_m00 = qi_m00[n];
    }
    if(p_000*p_p00<0) {
      s_p00_ = s_p00[n];
      q_p00 = qi_p00[n];
    }
    if(p_000*p_0m0<0) {
      s_0m0_ = s_0m0[n];
      q_0m0 = qi_0m0[n];
    }
    if(p_000*p_0p0<0){
      s_0p0_ = s_0p0[n];
      q_0p0 = qi_0p0[n];
    }
#ifdef P4_TO_P8
    if(p_000*p_00m<0){
      s_00m_ = s_00m[n];
      q_00m = qi_00m[n];
    }
    if(p_000*p_00p<0){
      s_00p_ = s_00p[n];
      q_00p = qi_00p[n];
    }
#endif

    double sgn = (p_000>0) ? 1 : -1;
    double qxx_000, qxx_m00, qxx_p00, qxx_0m0, qxx_0p0;
    double qyy_000, qyy_m00, qyy_p00, qyy_0m0, qyy_0p0;
#ifdef P4_TO_P8
    double qxx_00m, qxx_00p;
    double qyy_00m, qyy_00p;
    double qzz_000, qzz_m00, qzz_p00, qzz_0m0, qzz_0p0, qzz_00m, qzz_00p;
    qnnn.ngbd_with_quadratic_interpolation(qxx_p, qxx_000, qxx_m00, qxx_p00, qxx_0m0, qxx_0p0, qxx_00m, qxx_00p);
    qnnn.ngbd_with_quadratic_interpolation(qyy_p, qyy_000, qyy_m00, qyy_p00, qyy_0m0, qyy_0p0, qyy_00m, qyy_00p);
    qnnn.ngbd_with_quadratic_interpolation(qzz_p, qzz_000, qzz_m00, qzz_p00, qzz_0m0, qzz_0p0, qzz_00m, qzz_00p);
#else
    qnnn.ngbd_with_quadratic_interpolation(qxx_p, qxx_000, qxx_m00, qxx_p00, qxx_0m0, qxx_0p0);
    qnnn.ngbd_with_quadratic_interpolation(qyy_p, qyy_000, qyy_m00, qyy_p00, qyy_0m0, qyy_0p0);
#endif

    //---------------------------------------------------------------------
    // Neumann boundary condition on the walls
    //---------------------------------------------------------------------
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    /* wall in the x direction */
    if     (is_node_xmWall(p4est, node)) { s_m00_ = s_p00_; q_m00 = q_p00; qxx_000 = qxx_m00 = qxx_p00 = 0; }
    else if(is_node_xpWall(p4est, node)) { s_p00_ = s_m00_; q_p00 = q_m00; qxx_000 = qxx_m00 = qxx_p00 = 0; }

    /* wall in the y direction */
    if     (is_node_ymWall(p4est, node)) { s_0m0_ = s_0p0_; q_0m0 = q_0p0; qyy_000 = qyy_0m0 = qyy_0p0 = 0; }
    else if(is_node_ypWall(p4est, node)) { s_0p0_ = s_0m0_; q_0p0 = q_0m0; qyy_000 = qyy_0m0 = qyy_0p0 = 0; }

#ifdef P4_TO_P8
    /* wall in the y directin */
    if     (is_node_zmWall(p4est, node)) { s_00m_ = s_00p_; q_00m = q_00p; qzz_000 = qzz_00m = qzz_00p = 0; }
    else if(is_node_zpWall(p4est, node)) { s_00p_ = s_00m_; q_00p = q_00m; qzz_000 = qzz_00m = qzz_00p = 0; }
#endif

    //---------------------------------------------------------------------
    // Second order accurate One-Sided Differecing
    //---------------------------------------------------------------------
    double qxm = ((fabs(s_m00_) > EPS*qnnn.d_m00)? (q_000-q_m00)/s_m00_ : 0.0) + 0.5*s_m00_*MINMOD(qxx_m00, qxx_000);
    double qxp = ((fabs(s_p00_) > EPS*qnnn.d_p00)? (q_p00-q_000)/s_p00_ : 0.0) - 0.5*s_p00_*MINMOD(qxx_p00, qxx_000);
    double qym = ((fabs(s_0m0_) > EPS*qnnn.d_0m0)? (q_000-q_0m0)/s_0m0_ : 0.0) + 0.5*s_0m0_*MINMOD(qyy_0m0, qyy_000);
    double qyp = ((fabs(s_0p0_) > EPS*qnnn.d_0p0)? (q_0p0-q_000)/s_0p0_ : 0.0) - 0.5*s_0p0_*MINMOD(qyy_0p0, qyy_000);
#ifdef P4_TO_P8
    double qzm = ((fabs(s_00m_) > EPS*qnnn.d_00m)? (q_000-q_00m)/s_00m_ : 0.0) + 0.5*s_00m_*MINMOD(qzz_00m, qzz_000);
    double qzp = ((fabs(s_00p_) > EPS*qnnn.d_00p)? (q_00p-q_000)/s_00p_ : 0.0) - 0.5*s_00p_*MINMOD(qzz_00p, qzz_000);
#endif

    //---------------------------------------------------------------------
    // Upwind Scheme
    //---------------------------------------------------------------------
    double dt = MIN(s_m00_, s_p00_);
    dt = MIN(dt, s_0m0_, s_0p0_);
#ifdef P4_TO_P8
    dt = MIN(dt, s_00m_, s_00p_);
    dt /= 3.;
#else
    dt /= 2.;
#endif

    q_out_p[n] = q_000 - (dt*sgn) * ( nx[n]*( (sgn*nx[n]>0) ? qxm : qxp) +
                                      ny[n]*( (sgn*ny[n]>0) ? qym : qyp)
                                  #ifdef P4_TO_P8
                                      + nz[n]*( (sgn*nz[n]>0) ? qzm : qzp)
                                  #endif
                                      );

  }
}


void my_p4est_level_set_t::extend_from_interface_to_whole_domain_TVD( Vec phi, Vec qi, Vec q, int iterations ) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_from_interface_TVD, phi, qi, q, 0); CHKERRXX(ierr);

  /* init the neighborhood information if needed */
  /* NOTE: from now on the neighbors will be initialized ... do we want to clear them
   * at the end of this function if they were not initialized beforehand ?
   */
  ngbd->init_neighbors();

  /* find dx and dy smallest */
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  double dx = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  double dy = (ymax-ymin) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  double dz = (zmax-zmin) / pow(2.,(double) data->max_lvl);
  double dl = MAX(dx, dy, dz);
#else
  double dl = MAX(dx, dy);
#endif

  Vec qxx, qyy;
  double *qxx_p, *qyy_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qxx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec qzz; double *qzz_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qzz); CHKERRXX(ierr);
  compute_derivatives(qi, qxx, qyy, qzz);
#else
  compute_derivatives(qi, qxx, qyy);
#endif

  Vec q1, q2;
  double *q1_p, *q2_p, *q_p, *qi_p, *phi_p;
  ierr = VecDuplicate(phi, &q1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &q2); CHKERRXX(ierr);

  ierr = VecGetArray(qi, &qi_p); CHKERRXX(ierr);
  ierr = VecGetArray(q , &q_p); CHKERRXX(ierr);
  ierr = VecGetArray(q1, &q1_p); CHKERRXX(ierr);
  ierr = VecGetArray(q2, &q2_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  /* compute the normals */
  std::vector<double> nx(nodes->num_owned_indeps);
  std::vector<double> ny(nodes->num_owned_indeps);
#ifdef P4_TO_P8
  std::vector<double> nz(nodes->num_owned_indeps);
#endif
  quad_neighbor_nodes_of_node_t qnnn;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);
    nx[n] = qnnn.dx_central(phi_p);
    ny[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    nz[n] = qnnn.dz_central(phi_p);
    double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n] + nz[n]*nz[n]);
#else
    double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n]);
#endif

    if(norm>EPS)
    {
      nx[n] /= norm;
      ny[n] /= norm;
#ifdef P4_TO_P8
      nz[n] /= norm;
#endif
    }
    else
    {
      nx[n] = 0;
      ny[n] = 0;
#ifdef P4_TO_P8
      nz[n] = 0;
#endif
    }
  }

  /* compute second order derivatives of phi for second order accurate location */
  Vec dxx; double *dxx_p; ierr = VecCreateGhostNodes(p4est, nodes, &dxx); CHKERRXX(ierr);
  Vec dyy; double *dyy_p; ierr = VecCreateGhostNodes(p4est, nodes, &dyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec dzz; double *dzz_p; ierr = VecCreateGhostNodes(p4est, nodes, &dzz); CHKERRXX(ierr);
#endif

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  compute_derivatives(phi, dxx, dyy, dzz);
#else
  compute_derivatives(phi, dxx, dyy);
#endif

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecGetArray(dxx, &dxx_p); CHKERRXX(ierr);
  ierr = VecGetArray(dyy, &dyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecGetArray(dzz, &dzz_p); CHKERRXX(ierr);
#endif

  /* initialization of q */
  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    q_p[n] = fabs(phi_p[n])<1.5*dl ? qi_p[n] : 0;

  // first initialize the quantities at the interface (instead of doing it each time in the loop ...)
  std::vector<double> qi_m00(nodes->num_owned_indeps);
  std::vector<double> qi_p00(nodes->num_owned_indeps);
  std::vector<double> qi_0m0(nodes->num_owned_indeps);
  std::vector<double> qi_0p0(nodes->num_owned_indeps);
  std::vector<double> s_m00(nodes->num_owned_indeps);
  std::vector<double> s_p00(nodes->num_owned_indeps);
  std::vector<double> s_0m0(nodes->num_owned_indeps);
  std::vector<double> s_0p0(nodes->num_owned_indeps);

#ifdef P4_TO_P8
  std::vector<double> qi_00m(nodes->num_owned_indeps);
  std::vector<double> qi_00p(nodes->num_owned_indeps);
  std::vector<double> s_00m(nodes->num_owned_indeps);
  std::vector<double> s_00p(nodes->num_owned_indeps);
  my_p4est_interpolation_nodes_t interp_m00(ngbd); interp_m00.set_input(qi, qxx, qyy, qzz, interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_p00(ngbd); interp_p00.set_input(qi, qxx, qyy, qzz, interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_0m0(ngbd); interp_0m0.set_input(qi, qxx, qyy, qzz, interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_0p0(ngbd); interp_0p0.set_input(qi, qxx, qyy, qzz, interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_00m(ngbd); interp_00m.set_input(qi, qxx, qyy, qzz, interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_00p(ngbd); interp_00p.set_input(qi, qxx, qyy, qzz, interpolation_on_interface);
#else
  my_p4est_interpolation_nodes_t interp_m00(ngbd); interp_m00.set_input(qi, qxx, qyy, interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_p00(ngbd); interp_p00.set_input(qi, qxx, qyy, interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_0m0(ngbd); interp_0m0.set_input(qi, qxx, qyy, interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_0p0(ngbd); interp_0p0.set_input(qi, qxx, qyy, interpolation_on_interface);
#endif

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);
    double x = node_x_fr_n(n, p4est, nodes);
    double y = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z = node_z_fr_n(n, p4est, nodes);
#endif

    double p_000, p_m00, p_p00, p_0m0, p_0p0;
#ifdef P4_TO_P8
    double p_00m, p_00p;
#endif
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000,
                                                 p_m00, p_p00,
                                                 p_0m0, p_0p0
                                             #ifdef P4_TO_P8
                                                 , p_00m, p_00p
                                             #endif
                                                 );

    double s_p00_ = qnnn.d_p00; double s_m00_ = qnnn.d_m00;
    double s_0p0_ = qnnn.d_0p0; double s_0m0_ = qnnn.d_0m0;
#ifdef P4_TO_P8
    double s_00p_ = qnnn.d_00p; double s_00m_ = qnnn.d_00m;
#endif

    //---------------------------------------------------------------------
    // Second Order derivatives
    //---------------------------------------------------------------------
    double pxx_000 = dxx_p[n];
    double pyy_000 = dyy_p[n];
#ifdef P4_TO_P8
    double pzz_000 = dzz_p[n];
#endif
    double pxx_m00 = qnnn.f_m00_linear(dxx_p);
    double pxx_p00 = qnnn.f_p00_linear(dxx_p);
    double pyy_0m0 = qnnn.f_0m0_linear(dyy_p);
    double pyy_0p0 = qnnn.f_0p0_linear(dyy_p);
#ifdef P4_TO_P8
    double pzz_00m = qnnn.f_00m_linear(dzz_p);
    double pzz_00p = qnnn.f_00p_linear(dzz_p);
#endif

    if(p_000*p_m00<0){
//      s_m00[n] = interface_Location(0, s_m00_, p_000, p_m00);
      s_m00[n] =-interface_Location_With_Second_Order_Derivative(-s_m00_,   0,p_m00,p_000,pxx_m00,pxx_000);
      s_m00[n] = MAX(s_m00[n],EPS);
      double xyz[] = { x-s_m00[n], y
                 #ifdef P4_TO_P8
                       , z
                 #endif
                     };
      interp_m00.add_point(n, xyz);
    }
    else {
      qi_m00[n] = qi_p[n];
      s_m00[n] = s_m00_;
    }
    if(p_000*p_p00<0) {
//      s_p00[n] = interface_Location(0, s_p00_, p_000, p_p00);
      s_p00[n] = interface_Location_With_Second_Order_Derivative(    0,s_p00_,p_000,p_p00,pxx_000,pxx_p00);
      s_p00[n] = MAX(s_p00[n],EPS);
      double xyz[] = { x+s_p00[n], y
                 #ifdef P4_TO_P8
                       , z
                 #endif
                     };
      interp_p00.add_point(n, xyz);
    }
    else {
      qi_p00[n] = qi_p[n];
      s_p00[n] = s_p00_;
    }
    if(p_000*p_0m0<0) {
//      s_0m0[n] = interface_Location(0, s_0m0_, p_000, p_0m0);
      s_0m0[n] =-interface_Location_With_Second_Order_Derivative(-s_0m0_,   0,p_0m0,p_000,pyy_0m0,pyy_000);
      s_0m0[n] = MAX(s_0m0[n],EPS);
      double xyz[] = { x, y-s_0m0[n]
                 #ifdef P4_TO_P8
                       , z
                 #endif
                     };
      interp_0m0.add_point(n, xyz);
    }
    else {
      qi_0m0[n] = qi_p[n];
      s_0m0[n] = s_0m0_;
    }
    if(p_000*p_0p0<0){
//      s_0p0[n] = interface_Location(0, s_0p0_, p_000, p_0p0);
      s_0p0[n] = interface_Location_With_Second_Order_Derivative(    0,s_0p0_,p_000,p_0p0,pyy_000,pyy_0p0);
      s_0p0[n] = MAX(s_0p0[n],EPS);
      double xyz[] = { x, y+s_0p0[n]
                 #ifdef P4_TO_P8
                       , z
                 #endif
                     };
      interp_0p0.add_point(n, xyz);
    }
    else{
      qi_0p0[n] = qi_p[n];
      s_0p0[n] = s_0p0_;
    }
#ifdef P4_TO_P8
    if(p_000*p_00m<0) {
//      s_00m[n] = interface_Location(0, s_00m_, p_000, p_00m);
      s_00m[n] =-interface_Location_With_Second_Order_Derivative(-s_00m_,   0,p_00m,p_000,pzz_00m,pzz_000);
      s_00m[n] = MAX(s_00m[n],EPS);
      double xyz[] = { x, y, z-s_00m[n]};
      interp_00m.add_point(n, xyz);
    }
    else {
      qi_00m[n] = qi_p[n];
      s_00m[n] = s_00m_;
    }
    if(p_000*p_00p<0) {
//      s_00p[n] = interface_Location(0, s_00p_, p_000, p_00p);
      s_00p[n] = interface_Location_With_Second_Order_Derivative(    0,s_00p_,p_000,p_00p,pzz_000,pzz_00p);
      s_00p[n] = MAX(s_00p[n],EPS);
      double xyz[] = { x, y, z+s_00p[n] };
      interp_00p.add_point(n, xyz);
    }
    else {
      qi_00p[n] = qi_p[n];
      s_00p[n] = s_00p_;
    }
#endif
  }

  interp_m00.interpolate(qi_m00.data());
  interp_p00.interpolate(qi_p00.data());
  interp_0m0.interpolate(qi_0m0.data());
  interp_0p0.interpolate(qi_0p0.data());
#ifdef P4_TO_P8
  interp_00m.interpolate(qi_00m.data());
  interp_00p.interpolate(qi_00p.data());
#endif

  for(int it=0; it<iterations; ++it)
  {
    //---------------------------------------------------------------------
    // q1 = q - dt*sgn(phi)*n \cdot \nabla(q) by the Godunov scheme with ENO-2 and subcell resolution
    //---------------------------------------------------------------------

#ifdef P4_TO_P8
    compute_derivatives(q, qxx, qyy, qzz);
#else
    compute_derivatives(q, qxx, qyy);
#endif

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    /* First do layer nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration(layer_nodes, phi_p, nx, ny,
                                                        #ifdef P4_TO_P8
                                                            nz,
                                                        #endif
                                                            q1_p,
                                                            q_p, qxx_p, qyy_p,
                                                        #ifdef P4_TO_P8
                                                            qzz_p,
                                                        #endif
                                                            qi_m00, qi_p00,
                                                            qi_0m0, qi_0p0,
                                                        #ifdef P4_TO_P8
                                                            qi_00m, qi_00p,
                                                        #endif
                                                            s_m00, s_p00,
                                                            s_0m0, s_0p0
                                                        #ifdef P4_TO_P8
                                                            , s_00m, s_00p
                                                        #endif
                                                            );

    /* initiate communication for q1 */
    ierr = VecGhostUpdateBegin(q1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* compute the local nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration(local_nodes, phi_p, nx, ny,
                                                        #ifdef P4_TO_P8
                                                            nz,
                                                        #endif
                                                            q1_p,
                                                            q_p, qxx_p, qyy_p,
                                                        #ifdef P4_TO_P8
                                                            qzz_p,
                                                        #endif
                                                            qi_m00, qi_p00,
                                                            qi_0m0, qi_0p0,
                                                        #ifdef P4_TO_P8
                                                            qi_00m, qi_00p,
                                                        #endif
                                                            s_m00, s_p00,
                                                            s_0m0, s_0p0
                                                        #ifdef P4_TO_P8
                                                            , s_00m, s_00p
                                                        #endif
                                                            );

    /* finish communication for q1 */
    ierr = VecGhostUpdateEnd(q1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    compute_derivatives(q1, qxx, qyy, qzz);
#else
    compute_derivatives(q1, qxx, qyy);
#endif

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    //---------------------------------------------------------------------
    // q2 = q1 - dt*sgn(phi)*n \cdot \nabla(q1) by the Godunov scheme with ENO-2 and subcell resolution
    //---------------------------------------------------------------------
    /* First do layer nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration(layer_nodes, phi_p, nx, ny,
                                                        #ifdef P4_TO_P8
                                                            nz,
                                                        #endif
                                                            q2_p,
                                                            q1_p, qxx_p, qyy_p,
                                                        #ifdef P4_TO_P8
                                                            qzz_p,
                                                        #endif
                                                            qi_m00, qi_p00,
                                                            qi_0m0, qi_0p0,
                                                        #ifdef P4_TO_P8
                                                            qi_00m, qi_00p,
                                                        #endif
                                                            s_m00, s_p00,
                                                            s_0m0, s_0p0
                                                        #ifdef P4_TO_P8
                                                            , s_00m, s_00p
                                                        #endif
                                                            );

    /* initiate communication for q2 */
    ierr = VecGhostUpdateBegin(q2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* compute the local nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration(local_nodes, phi_p, nx, ny,
                                                        #ifdef P4_TO_P8
                                                            nz,
                                                        #endif
                                                            q2_p,
                                                            q1_p, qxx_p, qyy_p,
                                                        #ifdef P4_TO_P8
                                                            qzz_p,
                                                        #endif
                                                            qi_m00, qi_p00,
                                                            qi_0m0, qi_0p0,
                                                        #ifdef P4_TO_P8
                                                            qi_00m, qi_00p,
                                                        #endif
                                                            s_m00, s_p00,
                                                            s_0m0, s_0p0
                                                        #ifdef P4_TO_P8
                                                            , s_00m, s_00p
                                                        #endif
                                                            );

    /* finish communication for q2 */
    ierr = VecGhostUpdateEnd(q2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    //---------------------------------------------------------------------
    // The third step of TVD RK-2 : q = .5*(q + q2)
    //---------------------------------------------------------------------
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      q_p[n] = .5*(q_p[n] + q2_p[n]);
  }

  /* destroy the local petsc vectors */

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(qi, &qi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(q , &q_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(q1, &q1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(q2, &q2_p); CHKERRXX(ierr);

  ierr = VecDestroy(q1); CHKERRXX(ierr);
  ierr = VecDestroy(q2); CHKERRXX(ierr);
  ierr = VecDestroy(qxx); CHKERRXX(ierr);
  ierr = VecDestroy(qyy); CHKERRXX(ierr);

#ifdef P4_TO_P8
  ierr = VecDestroy(qzz); CHKERRXX(ierr);
#endif

  ierr = VecRestoreArray(dxx, &dxx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(dyy, &dyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(dzz, &dzz_p); CHKERRXX(ierr);
#endif

  ierr = VecDestroy(dxx);  CHKERRXX(ierr);
  ierr = VecDestroy(dyy);  CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(dzz); CHKERRXX(ierr);
#endif

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_from_interface_TVD, phi, qi, q, 0); CHKERRXX(ierr);
}


void my_p4est_level_set_t::enforce_contact_angle(Vec phi_wall, Vec phi_intf, Vec cos_angle, int iterations, Vec normal[]) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_over_interface_TVD, phi_wall, phi_intf, 0, 0); CHKERRXX(ierr);

  double *phi_wall_p;

  ierr = VecGetArray(phi_wall, &phi_wall_p); CHKERRXX(ierr);

  Vec tmp, tmp_loc, q_loc;
  ierr = VecDuplicate(phi_wall, &tmp); CHKERRXX(ierr);
  double *tmp_p;

  double dxyz[P4EST_DIM];

  dxyz_min(p4est, dxyz);

  /* init the neighborhood information if needed */
  ngbd->init_neighbors();
  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  /* compute the normals */
  double *normal_p[P4EST_DIM];

  if (normal != NULL)
    for (short dim = 0; dim < P4EST_DIM; ++dim)
    {
      ierr = VecGetArray(normal[dim], &normal_p[dim]); CHKERRXX(ierr);
    }

  std::vector<double> nx(nodes->num_owned_indeps);
  std::vector<double> ny(nodes->num_owned_indeps);
#ifdef P4_TO_P8
  std::vector<double> nz(nodes->num_owned_indeps);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  if (normal != NULL) {
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      nx[n] = normal_p[0][n];
      ny[n] = normal_p[1][n];
#ifdef P4_TO_P8
      nz[n] = normal_p[2][n];
#endif
    }
  } else {
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      ngbd->get_neighbors(n, qnnn);
      nx[n] = qnnn.dx_central(phi_wall_p);
      ny[n] = qnnn.dy_central(phi_wall_p);
#ifdef P4_TO_P8
      nz[n] = qnnn.dz_central(phi_wall_p);
      double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n] + nz[n]*nz[n]);
#else
      double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n]);
#endif

      if(norm>EPS)
      {
        nx[n] /= norm;
        ny[n] /= norm;
#ifdef P4_TO_P8
        nz[n] /= norm;
#endif
      }
      else
      {
        nx[n] = 0;
        ny[n] = 0;
#ifdef P4_TO_P8
        nz[n] = 0;
#endif
      }
    }
  }

  if (normal != NULL)
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArray(normal[dim], &normal_p[dim]); CHKERRXX(ierr);
    }

  /* extrapolate q */
  double *q_p;

  Vec qxx, qyy;
  double *qxx_p, *qyy_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qxx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec qzz;
  double *qzz_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qzz); CHKERRXX(ierr);
#endif

  double *cos_angle_p;
  ierr = VecGetArray(cos_angle, &cos_angle_p); CHKERRXX(ierr);

  for(int it=0; it<iterations; ++it)
  {
#ifdef P4_TO_P8
    ngbd->second_derivatives_central(phi_intf, qxx, qyy, qzz);
#else
    ngbd->second_derivatives_central(phi_intf, qxx, qyy);
#endif

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(phi_intf, &q_p); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_wall_p[n] > EPS)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
        dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
        dt /= 3;
#else
        dt /= 2;
#endif

        /* first order one sided derivatives */
        double qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                            : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                            : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                            : (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        /* second order derivatives */
        double qxx_m00 = qnnn.f_m00_linear(qxx_p);
        double qxx_p00 = qnnn.f_p00_linear(qxx_p);
        double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
        double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
        double qzz_00m = qnnn.f_00m_linear(qzz_p);
        double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

        /* minmod operation */
        qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
        else        qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
        else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

        double qx_m00 = (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00;
        double qx_p00 = (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy_0m0 = (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0;
        double qy_0p0 = (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz_00m = (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m;
        double qz_00p = (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        qx_m00 += .5*qnnn.d_m00*qxx_m00;
        qx_p00 -= .5*qnnn.d_p00*qxx_p00;
        qy_0m0 += .5*qnnn.d_0m0*qyy_0m0;
        qy_0p0 -= .5*qnnn.d_0p0*qyy_0p0;
#ifdef P4_TO_P8
        qz_00m += .5*qnnn.d_00m*qzz_00m;
        qz_00p -= .5*qnnn.d_00p*qzz_00p;
#endif

//        if(qx_p00>0) qx_p00 = 0;
//        if(qx_m00<0) qx_m00 = 0;
//        if(qy_0p0>0) qy_0p0 = 0;
//        if(qy_0m0<0) qy_0m0 = 0;
//#ifdef P4_TO_P8
//        if(qz_00p>0) qz_00p = 0;
//        if(qz_00m<0) qz_00m = 0;
//#endif
        if(qx_p00<0) qx_p00 = 0;
        if(qx_m00>0) qx_m00 = 0;
        if(qy_0p0<0) qy_0p0 = 0;
        if(qy_0m0>0) qy_0m0 = 0;
#ifdef P4_TO_P8
        if(qz_00p<0) qz_00p = 0;
        if(qz_00m>0) qz_00m = 0;
#endif

        tmp_p[n] = q_p[n] - dt*( nx[n]*qx + ny[n]*qy
                         #ifdef P4_TO_P8
                                 + nz[n]*qz
                         #endif
                                 - cos_angle_p[n]*sqrt( MAX(qx_p00*qx_p00 , qx_m00*qx_m00) +
                                                        MAX(qy_0p0*qy_0p0 , qy_0m0*qy_0m0) ));
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_wall_p[n] > -EPS)
      {
        double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
        dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
        dt /= 3;
#else
        dt /= 2;
#endif

        /* first order one sided derivatives */
        double qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                            : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                            : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                            : (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        /* second order derivatives */
        double qxx_m00 = qnnn.f_m00_linear(qxx_p);
        double qxx_p00 = qnnn.f_p00_linear(qxx_p);
        double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
        double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
        double qzz_00m = qnnn.f_00m_linear(qzz_p);
        double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

        /* minmod operation */
        qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
        else        qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
        else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

        double qx_m00 = (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00;
        double qx_p00 = (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy_0m0 = (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0;
        double qy_0p0 = (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz_00m = (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m;
        double qz_00p = (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        qx_m00 += .5*qnnn.d_m00*qxx_m00;
        qx_p00 -= .5*qnnn.d_p00*qxx_p00;
        qy_0m0 += .5*qnnn.d_0m0*qyy_0m0;
        qy_0p0 -= .5*qnnn.d_0p0*qyy_0p0;
#ifdef P4_TO_P8
        qz_00m += .5*qnnn.d_00m*qzz_00m;
        qz_00p -= .5*qnnn.d_00p*qzz_00p;
#endif

//        if(qx_p00>0) qx_p00 = 0;
//        if(qx_m00<0) qx_m00 = 0;
//        if(qy_0p0>0) qy_0p0 = 0;
//        if(qy_0m0<0) qy_0m0 = 0;
//#ifdef P4_TO_P8
//        if(qz_00p>0) qz_00p = 0;
//        if(qz_00m<0) qz_00m = 0;
//#endif
//        if(qx_p00<0) qx_p00 = 0;
//        if(qx_m00>0) qx_m00 = 0;
//        if(qy_0p0<0) qy_0p0 = 0;
//        if(qy_0m0>0) qy_0m0 = 0;
//#ifdef P4_TO_P8
//        if(qz_00p<0) qz_00p = 0;
//        if(qz_00m>0) qz_00m = 0;
//#endif

        double abs_grad_q = sqrt( MAX(qx_p00*qx_p00 , qx_m00*qx_m00) +
                                  MAX(qy_0p0*qy_0p0 , qy_0m0*qy_0m0) );

        abs_grad_q = 1;

        tmp_p[n] = q_p[n] - dt*( nx[n]*qx + ny[n]*qy
                         #ifdef P4_TO_P8
                                 + nz[n]*qz
                         #endif
                                 - cos_angle_p[n]*abs_grad_q);
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(phi_intf , &q_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

    ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(phi_intf, &q_loc); CHKERRXX(ierr);
    ierr = VecCopy(tmp_loc, q_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi_intf, &q_loc); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(phi_wall, &phi_wall_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(cos_angle, &cos_angle_p); CHKERRXX(ierr);

  ierr = VecDestroy(qxx); CHKERRXX(ierr);
  ierr = VecDestroy(qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(qzz); CHKERRXX(ierr);
#endif

  ierr = VecDestroy(tmp); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_over_interface_TVD, phi_wall, phi_intf, 0, 0); CHKERRXX(ierr);
}


void my_p4est_level_set_t::enforce_contact_angle2(Vec phi, Vec q, Vec cos_angle, int iterations, int order, Vec normal[]) const
{
#ifdef CASL_THROWS
  if(order!=0 && order!=1 && order!=2) throw std::invalid_argument("[CASL_ERROR]: my_p4est_level_set_t->extend_Over_Interface_TVD: order must be 0, 1 or 2.");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_over_interface_TVD, phi, q, 0, 0); CHKERRXX(ierr);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  Vec qn, qnn;
  double *q_p, *qn_p, *qnn_p;
  Vec b_qn_well_defined;
  Vec b_qnn_well_defined;
  double *b_qn_well_defined_p;
  double *b_qnn_well_defined_p;

  Vec tmp, tmp_loc;
  Vec qnn_loc, qn_loc, q_loc;
  ierr = VecDuplicate(phi, &tmp); CHKERRXX(ierr);
  double *tmp_p;

  double dxyz[P4EST_DIM];

  dxyz_min(p4est, dxyz);

  /* init the neighborhood information if needed */
  /* NOTE: from now on the neighbors will be initialized ... do we want to clear them
   * at the end of this function if they were not initialized beforehand ?
   */
  ngbd->init_neighbors();

  /* compute the normals */
  double *normal_p[P4EST_DIM];

  if (normal != NULL)
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecGetArray(normal[dim], &normal_p[dim]); CHKERRXX(ierr);
    }

  std::vector<double> nx(nodes->num_owned_indeps);
  std::vector<double> ny(nodes->num_owned_indeps);
#ifdef P4_TO_P8
  std::vector<double> nz(nodes->num_owned_indeps);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  if (normal != NULL) {
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      nx[n] = normal_p[0][n];
      ny[n] = normal_p[1][n];
#ifdef P4_TO_P8
      nz[n] = normal_p[2][n];
#endif
    }
  } else {
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    {
      ngbd->get_neighbors(n, qnnn);
      nx[n] = qnnn.dx_central(phi_p);
      ny[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
      nz[n] = qnnn.dz_central(phi_p);
      double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n] + nz[n]*nz[n]);
#else
      double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n]);
#endif

      if(norm>EPS)
      {
        nx[n] /= norm;
        ny[n] /= norm;
#ifdef P4_TO_P8
        nz[n] /= norm;
#endif
      }
      else
      {
        nx[n] = 0;
        ny[n] = 0;
#ifdef P4_TO_P8
        nz[n] = 0;
#endif
      }
    }
  }

  if (normal != NULL)
    for (short dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArray(normal[dim], &normal_p[dim]); CHKERRXX(ierr);
    }

  ierr = VecGetArray(q , &q_p) ; CHKERRXX(ierr);

  /* initialize qn */
  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  if(order >=1 )
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &qn); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &b_qn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  phi_p[qnnn.node_000]<-EPS &&
     #ifdef P4_TO_P8
           ( phi_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( phi_p[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( phi_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( phi_p[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( phi_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( phi_p[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( phi_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( phi_p[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( phi_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( phi_p[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( phi_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( phi_p[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( phi_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( phi_p[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( phi_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( phi_p[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( phi_p[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( phi_p[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( phi_p[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( phi_p[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( phi_p[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( phi_p[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( phi_p[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( phi_p[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( phi_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
           ( phi_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
           ( phi_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
           ( phi_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
           ( phi_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( phi_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( phi_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( phi_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        qn_p[n] = ( nx[n]*qnnn.dx_central(q_p) +
                    ny[n]*qnnn.dy_central(q_p)
            #ifdef P4_TO_P8
                    + nz[n]*qnnn.dz_central(q_p)
            #endif
                    );
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  phi_p[qnnn.node_000]<-EPS &&
     #ifdef P4_TO_P8
           ( phi_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( phi_p[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( phi_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( phi_p[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( phi_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( phi_p[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( phi_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( phi_p[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( phi_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( phi_p[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( phi_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( phi_p[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( phi_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( phi_p[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( phi_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( phi_p[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( phi_p[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( phi_p[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( phi_p[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( phi_p[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( phi_p[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( phi_p[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( phi_p[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( phi_p[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( phi_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
           ( phi_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
           ( phi_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
           ( phi_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
           ( phi_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( phi_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( phi_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( phi_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        qn_p[n] = ( nx[n]*qnnn.dx_central(q_p) +
                    ny[n]*qnnn.dy_central(q_p)
            #ifdef P4_TO_P8
                    + nz[n]*qnnn.dz_central(q_p)
            #endif
                    );
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0;
      }
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);

  /* initialize qnn */
  if(order == 2)
  {
    ierr = VecDuplicate(qn, &qnn); CHKERRXX(ierr);
    ierr = VecDuplicate(b_qn_well_defined, &b_qnn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecGetArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000]==true &&
     #ifdef P4_TO_P8
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_mp]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pp]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mp]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pp]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mp]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pp]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mp]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pp]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00m_mm]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_mp]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pm]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pp]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00p_mm]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_mp]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pm]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pp]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        qnn_p[n] = ( nx[n]*qnnn.dx_central(qn_p) +
                     ny[n]*qnnn.dy_central(qn_p)
             #ifdef P4_TO_P8
                     + nz[n]*qnnn.dz_central(qn_p)
             #endif
                     );
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000]==true &&
     #ifdef P4_TO_P8
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_mp]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pp]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mp]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pp]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mp]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pp]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mp]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pp]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00m_mm]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_mp]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pm]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pp]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00p_mm]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_mp]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pm]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pp]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        qnn_p[n] = ( nx[n]*qnnn.dx_central(qn_p) +
                     ny[n]*qnnn.dy_central(qn_p)
             #ifdef P4_TO_P8
                     + nz[n]*qnnn.dz_central(qn_p)
             #endif
                     );
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0;
      }
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  /* extrapolate qnn */
  if(order==2)
  {
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    for(int it=0; it<iterations; ++it)
    {
      ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnnx = nx[n]>0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n]>0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n]>0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qnn_p[n] - dt*( nx[n]*qnnx + ny[n]*qnny
                           #ifdef P4_TO_P8
                                     + nz[n]*qnnz
                           #endif
                                     );
        }
        else
          tmp_p[n] = qnn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnnx = nx[n]>0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n]>0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n]>0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qnn_p[n] - dt*( nx[n]*qnnx + ny[n]*qnny
                           #ifdef P4_TO_P8
                                     + nz[n]*qnnz
                           #endif
                                     );
        }
        else
          tmp_p[n] = qnn_p[n];
      }

      /* end update communication */
      ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

      ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(qnn, &qnn_loc); CHKERRXX(ierr);
      ierr = VecCopy(tmp_loc, qnn_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(qnn, &qnn_loc); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  /* extrapolate qn */
  if(order>=1)
  {
    double *cos_angle_p;

    ierr = VecGetArray(cos_angle, &cos_angle_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);

    foreach_node(n, nodes)
    {
      qn_p[n] = cos_angle_p[n];
      if (phi_p[n] < 0)
      {
        b_qn_well_defined_p[n] = true;
      } else {
        b_qn_well_defined_p[n] = false;
      }
    }

    ierr = VecRestoreArray(cos_angle, &cos_angle_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);

    if(order==2) ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for(int it=0; it<iterations; ++it)
    {
      ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnx = nx[n]>0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                               : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n]>0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                               : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n]>0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                               : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qn_p[n] - dt*( nx[n]*qnx + ny[n]*qny
                          #ifdef P4_TO_P8
                                    + nz[n]*qnz
                          #endif
                                    ) + (order==2 ? dt*qnn_p[n] : 0);
        }
        else
          tmp_p[n] = qn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnx = nx[n]>0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                               : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n]>0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                               : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n]>0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                               : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qn_p[n] - dt*( nx[n]*qnx + ny[n]*qny
                          #ifdef P4_TO_P8
                                    + nz[n]*qnz
                          #endif
                                    ) + (order==2 ? dt*qnn_p[n] : 0);
        }
        else
          tmp_p[n] = qn_p[n];
      }

      /* end update communication */
      ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

      ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(qn , &qn_loc ); CHKERRXX(ierr);
      ierr = VecCopy(tmp_loc, qn_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(qn , &qn_loc ); CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    if(order==2)
    {
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecDestroy(qnn); CHKERRXX(ierr);
    }
  }

  if(order>=1) { ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr); }
  if(order==2) { ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr); }

  /* extrapolate q */
  Vec qxx, qyy;
  double *qxx_p, *qyy_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qxx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec qzz;
  double *qzz_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qzz); CHKERRXX(ierr);
#endif

  if(order>=1) { ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr); }

  for(int it=0; it<iterations; ++it)
  {
#ifdef P4_TO_P8
    ngbd->second_derivatives_central(q, qxx, qyy, qzz);
#else
    ngbd->second_derivatives_central(q, qxx, qyy);
#endif

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_p[n] > -EPS)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
        dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
        dt /= 3;
#else
        dt /= 2;
#endif

        /* first order one sided derivatives */
        double qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                            : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                            : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                            : (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        /* second order derivatives */
        double qxx_m00 = qnnn.f_m00_linear(qxx_p);
        double qxx_p00 = qnnn.f_p00_linear(qxx_p);
        double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
        double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
        double qzz_00m = qnnn.f_00m_linear(qzz_p);
        double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

        /* minmod operation */
        qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
        else        qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
        else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

//#ifdef P4_TO_P8
//        if(fabs(nx[n])<EPS && fabs(ny[n])<EPS && fabs(nz[n])<EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p) +
//                      qnnn.f_00m_linear(q_p) + qnnn.f_00p_linear(q_p))/6.;
//#else
//        if(fabs(nx[n])<EPS && fabs(ny[n])<EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p))/4.;
//#endif
//        else
          tmp_p[n] = q_p[n] - dt*( nx[n]*qx + ny[n]*qy
                         #ifdef P4_TO_P8
                                   + nz[n]*qz
                         #endif
                                   ) + (order>=1 ? dt*qn_p[n] : 0);
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_p[n] > -EPS)
      {
        double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
        dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
        dt /= 3;
#else
        dt /= 2;
#endif

        /* first order one sided derivatives */
        double qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                            : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                            : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                            : (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        /* second order derivatives */
        double qxx_m00 = qnnn.f_m00_linear(qxx_p);
        double qxx_p00 = qnnn.f_p00_linear(qxx_p);
        double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
        double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
        double qzz_00m = qnnn.f_00m_linear(qzz_p);
        double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

        /* minmod operation */
        qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
        else        qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
        else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

//#ifdef P4_TO_P8
//        if(fabs(nx[n])<EPS && fabs(ny[n])<EPS && fabs(nz[n])<EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p) +
//                      qnnn.f_00m_linear(q_p) + qnnn.f_00p_linear(q_p))/6.;
//#else
//        if(fabs(nx[n])<EPS && fabs(ny[n])<EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p))/4.;
//#endif
//        else
          tmp_p[n] = q_p[n] - dt*( nx[n]*qx + ny[n]*qy
                         #ifdef P4_TO_P8
                                   + nz[n]*qz
                         #endif
                                   ) + (order>=1 ? dt*qn_p[n] : 0);
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

    ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(q  , &q_loc  ); CHKERRXX(ierr);
    ierr = VecCopy(tmp_loc, q_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(q  , &q_loc  ); CHKERRXX(ierr);
  }

  if(order>=1)
  {
    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecDestroy(qn); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = VecDestroy(qxx); CHKERRXX(ierr);
  ierr = VecDestroy(qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(qzz); CHKERRXX(ierr);
#endif

  ierr = VecDestroy(tmp); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_over_interface_TVD, phi, q, 0, 0); CHKERRXX(ierr);
}

double my_p4est_level_set_t::advect_in_normal_direction_with_contact_angle(const Vec vn, const Vec surf_tns, const Vec cos_angle, const Vec phi_wall, Vec phi, double dt)
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_advect_in_normal_direction_Vec, 0, 0, 0, 0);

  Vec rhs;
  ierr = VecDuplicate(phi, &rhs); CHKERRXX(ierr);

  // advection step using (assuming |grad(phi)| = 1)
  VecCopyGhost(phi, rhs);
  VecAXPBYGhost(rhs, -1, 1./dt, vn);

  // diffusion step (assuming |grad(phi)| = 1)
  my_p4est_interpolation_nodes_t interp(ngbd);
  Vec flux;
  ierr = VecDuplicate(phi, &flux); CHKERRXX(ierr);
  interp.set_input(flux, linear);

  my_p4est_poisson_nodes_mls_t solver(ngbd);
  if (phi_wall != NULL && cos_angle != NULL && use_neumann_for_contact_angle)
  {
    VecCopyGhost(cos_angle, flux);
    VecPointwiseMultGhost(flux, flux, surf_tns);

    solver.add_boundary(MLS_INTERSECTION, phi_wall, NULL, NEUMANN, interp, zero_cf);
  }

  solver.set_use_sc_scheme(0);
  solver.set_integration_order(1);

  solver.set_mu(surf_tns);
  solver.set_diag(1./dt);
  solver.set_rhs(rhs);

  solver.set_wc(bc_wall_type, zero_cf);

  solver.solve(phi, true);

  ierr = VecDestroy(rhs); CHKERRXX(ierr);
  ierr = VecDestroy(flux); CHKERRXX(ierr);

  // extend into wall
  if (phi_wall != NULL && cos_angle != NULL)
  {
    switch (contact_angle_extension)
    {
      case 0: enforce_contact_angle(phi_wall, phi, cos_angle, 50); break;
      case 1: enforce_contact_angle2(phi_wall, phi, cos_angle, 50); break;
      case 2:
        {
          Vec region;
          ierr = VecDuplicate(phi, &region); CHKERRXX(ierr);

          double *region_ptr;
          double *phi_ptr;
          double *phi_wall_ptr;

          ierr = VecGetArray(phi, &phi_ptr);           CHKERRXX(ierr);
          ierr = VecGetArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
          ierr = VecGetArray(region, &region_ptr);     CHKERRXX(ierr);

          double dxyz[P4EST_DIM];
          dxyz_min(p4est, dxyz);

#ifdef P4_TO_P8
          double dx = MIN(dxyz[0], dxyz[1], dxyz[2]);
#else
          double dx = MIN(dxyz[0], dxyz[1]);
#endif

          double limit_extd = 6.*dx;
          double limit_wall = 6.*dx;

          foreach_node(n, nodes)
          {
            if (fabs(phi_ptr[n]) < limit_extd && phi_wall_ptr[n] < limit_wall)
              region_ptr[n] = 1;
            else
              region_ptr[n] = 0;
          }

          ierr = VecRestoreArray(phi, &phi_ptr);           CHKERRXX(ierr);
          ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
          ierr = VecRestoreArray(region, &region_ptr);     CHKERRXX(ierr);

          extend_Over_Interface_TVD_regional(phi_wall, phi_wall, region, phi, 50, 2); break;
        }
    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_level_set_advect_in_normal_direction_Vec, 0, 0, 0, 0);
}

void my_p4est_level_set_t::extend_Over_Interface_TVD_regional( Vec phi, Vec mask, Vec region, Vec q, int iterations, int order) const
{
#ifdef CASL_THROWS
  if(order!=0 && order!=1 && order!=2) throw std::invalid_argument("[CASL_ERROR]: my_p4est_level_set_t->extend_Over_Interface_TVD: order must be 0, 1 or 2.");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_over_interface_TVD, phi, q, 0, 0); CHKERRXX(ierr);

  double *mask_p;
  ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  Vec qn, qnn;
  double *q_p, *qn_p, *qnn_p;
  Vec b_qn_well_defined;
  Vec b_qnn_well_defined;
  double *b_qn_well_defined_p;
  double *b_qnn_well_defined_p;

  Vec tmp, tmp_loc;
  Vec qnn_loc, qn_loc, q_loc;
  ierr = VecDuplicate(phi, &tmp); CHKERRXX(ierr);
  double *tmp_p;

  double dxyz[P4EST_DIM];

  dxyz_min(p4est, dxyz);


  /* init the neighborhood information if needed */
  /* NOTE: from now on the neighbors will be initialized ... do we want to clear them
   * at the end of this function if they were not initialized beforehand ?
   */
  ngbd->init_neighbors();

  /* compute the normals */
  std::vector<double> nx(nodes->num_owned_indeps);
  std::vector<double> ny(nodes->num_owned_indeps);
#ifdef P4_TO_P8
  std::vector<double> nz(nodes->num_owned_indeps);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);
    nx[n] = qnnn.dx_central(phi_p);
    ny[n] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
    nz[n] = qnnn.dz_central(phi_p);
    double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n] + nz[n]*nz[n]);
#else
    double norm = sqrt(nx[n]*nx[n] + ny[n]*ny[n]);
#endif

		if(norm>EPS)
		{
			nx[n] /= norm;
			ny[n] /= norm;
#ifdef P4_TO_P8
			nz[n] /= norm;
#endif
		}
		else
		{
			nx[n] = 0;
			ny[n] = 0;
#ifdef P4_TO_P8
			nz[n] = 0;
#endif
                }
  }

  ierr = VecGetArray(q , &q_p) ; CHKERRXX(ierr);

  /* initialize qn */
  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  if(order >=1 )
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &qn); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &b_qn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  mask_p[qnnn.node_000]<-EPS &&
     #ifdef P4_TO_P8
           ( mask_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_p[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( mask_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_p[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( mask_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_p[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( mask_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_p[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( mask_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_p[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( mask_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_p[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( mask_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_p[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( mask_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_p[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( mask_p[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_p[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( mask_p[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_p[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( mask_p[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_p[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( mask_p[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_p[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( mask_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
           ( mask_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
           ( mask_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
           ( mask_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
           ( mask_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( mask_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( mask_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( mask_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        qn_p[n] = ( nx[n]*qnnn.dx_central(q_p) +
                    ny[n]*qnnn.dy_central(q_p)
            #ifdef P4_TO_P8
                    + nz[n]*qnnn.dz_central(q_p)
            #endif
                    );
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  mask_p[qnnn.node_000]<-EPS &&
     #ifdef P4_TO_P8
           ( mask_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_p[qnnn.node_m00_mp]<-EPS || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( mask_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( mask_p[qnnn.node_m00_pp]<-EPS || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( mask_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_p[qnnn.node_p00_mp]<-EPS || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( mask_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( mask_p[qnnn.node_p00_pp]<-EPS || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( mask_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_p[qnnn.node_0m0_mp]<-EPS || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( mask_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( mask_p[qnnn.node_0m0_pp]<-EPS || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( mask_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_p[qnnn.node_0p0_mp]<-EPS || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( mask_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( mask_p[qnnn.node_0p0_pp]<-EPS || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( mask_p[qnnn.node_00m_mm]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_p[qnnn.node_00m_mp]<-EPS || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( mask_p[qnnn.node_00m_pm]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( mask_p[qnnn.node_00m_pp]<-EPS || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( mask_p[qnnn.node_00p_mm]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_p[qnnn.node_00p_mp]<-EPS || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( mask_p[qnnn.node_00p_pm]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( mask_p[qnnn.node_00p_pp]<-EPS || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( mask_p[qnnn.node_m00_mm]<-EPS || fabs(qnnn.d_m00_p0)<EPS) &&
           ( mask_p[qnnn.node_m00_pm]<-EPS || fabs(qnnn.d_m00_m0)<EPS) &&
           ( mask_p[qnnn.node_p00_mm]<-EPS || fabs(qnnn.d_p00_p0)<EPS) &&
           ( mask_p[qnnn.node_p00_pm]<-EPS || fabs(qnnn.d_p00_m0)<EPS) &&
           ( mask_p[qnnn.node_0m0_mm]<-EPS || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( mask_p[qnnn.node_0m0_pm]<-EPS || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( mask_p[qnnn.node_0p0_mm]<-EPS || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( mask_p[qnnn.node_0p0_pm]<-EPS || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        qn_p[n] = ( nx[n]*qnnn.dx_central(q_p) +
                    ny[n]*qnnn.dy_central(q_p)
            #ifdef P4_TO_P8
                    + nz[n]*qnnn.dz_central(q_p)
            #endif
                    );
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0;
      }
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);

  /* initialize qnn */
  if(order == 2)
  {
    ierr = VecDuplicate(qn, &qnn); CHKERRXX(ierr);
    ierr = VecDuplicate(b_qn_well_defined, &b_qnn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecGetArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000]==true &&
     #ifdef P4_TO_P8
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_mp]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pp]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mp]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pp]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mp]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pp]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mp]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pp]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00m_mm]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_mp]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pm]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pp]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00p_mm]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_mp]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pm]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pp]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        qnn_p[n] = ( nx[n]*qnnn.dx_central(qn_p) +
                     ny[n]*qnnn.dy_central(qn_p)
             #ifdef P4_TO_P8
                     + nz[n]*qnnn.dz_central(qn_p)
             #endif
                     );
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000]==true &&
     #ifdef P4_TO_P8
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_mp]==true || fabs(qnnn.d_m00_p0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pp]==true || fabs(qnnn.d_m00_m0)<EPS || fabs(qnnn.d_m00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mp]==true || fabs(qnnn.d_p00_p0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pp]==true || fabs(qnnn.d_p00_m0)<EPS || fabs(qnnn.d_p00_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mp]==true || fabs(qnnn.d_0m0_p0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pp]==true || fabs(qnnn.d_0m0_m0)<EPS || fabs(qnnn.d_0m0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mp]==true || fabs(qnnn.d_0p0_p0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pp]==true || fabs(qnnn.d_0p0_m0)<EPS || fabs(qnnn.d_0p0_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00m_mm]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_mp]==true || fabs(qnnn.d_00m_p0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pm]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pp]==true || fabs(qnnn.d_00m_m0)<EPS || fabs(qnnn.d_00m_0m)<EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00p_mm]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_mp]==true || fabs(qnnn.d_00p_p0)<EPS || fabs(qnnn.d_00p_0m)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pm]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0p)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pp]==true || fabs(qnnn.d_00p_m0)<EPS || fabs(qnnn.d_00p_0m)<EPS)
     #else
           ( b_qn_well_defined_p[qnnn.node_m00_mm]==true || fabs(qnnn.d_m00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm]==true || fabs(qnnn.d_m00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mm]==true || fabs(qnnn.d_p00_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm]==true || fabs(qnnn.d_p00_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mm]==true || fabs(qnnn.d_0m0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm]==true || fabs(qnnn.d_0m0_m0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mm]==true || fabs(qnnn.d_0p0_p0)<EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm]==true || fabs(qnnn.d_0p0_m0)<EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        qnn_p[n] = ( nx[n]*qnnn.dx_central(qn_p) +
                     ny[n]*qnnn.dy_central(qn_p)
             #ifdef P4_TO_P8
                     + nz[n]*qnnn.dz_central(qn_p)
             #endif
                     );
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0;
      }
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  /* extrapolate qnn */
  if(order==2)
  {
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    for(int it=0; it<iterations; ++it)
    {
      ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnnx = nx[n]>0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n]>0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n]>0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qnn_p[n] - dt*( nx[n]*qnnx + ny[n]*qnny
                           #ifdef P4_TO_P8
                                     + nz[n]*qnnz
                           #endif
                                     );
        }
        else
          tmp_p[n] = qnn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnnx = nx[n]>0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n]>0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n]>0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qnn_p[n] - dt*( nx[n]*qnnx + ny[n]*qnny
                           #ifdef P4_TO_P8
                                     + nz[n]*qnnz
                           #endif
                                     );
        }
        else
          tmp_p[n] = qnn_p[n];
      }

      /* end update communication */
      ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

      ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(qnn, &qnn_loc); CHKERRXX(ierr);
      ierr = VecCopy(tmp_loc, qnn_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(qnn, &qnn_loc); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  double *region_p;
  ierr = VecGetArray(region, &region_p); CHKERRXX(ierr);

  /* extrapolate qn */
  if(order>=1)
  {
    if(order==2) ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for(int it=0; it<iterations; ++it)
    {
      ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnx = nx[n]>0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                               : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n]>0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                               : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n]>0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                               : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qn_p[n] - dt*( nx[n]*qnx + ny[n]*qny
                          #ifdef P4_TO_P8
                                    + nz[n]*qnz
                          #endif
                                    ) + (order==2 ? dt*qnn_p[n]*region_p[n] : 0);
        }
        else
          tmp_p[n] = qn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
          dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
          dt /= 3;
#else
          dt /= 2;
#endif

          /* first order one sided derivative */
          double qnx = nx[n]>0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                               : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n]>0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                               : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n]>0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                               : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qn_p[n] - dt*( nx[n]*qnx + ny[n]*qny
                          #ifdef P4_TO_P8
                                    + nz[n]*qnz
                          #endif
                                    ) + (order==2 ? dt*qnn_p[n]*region_p[n] : 0);
        }
        else
          tmp_p[n] = qn_p[n];
      }

      /* end update communication */
      ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      ierr = VecRestoreArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

      ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostGetLocalForm(qn , &qn_loc ); CHKERRXX(ierr);
      ierr = VecCopy(tmp_loc, qn_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
      ierr = VecGhostRestoreLocalForm(qn , &qn_loc ); CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    if(order==2)
    {
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecDestroy(qnn); CHKERRXX(ierr);
    }
  }

  if(order>=1) { ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr); }
  if(order==2) { ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr); }

  /* extrapolate q */
  Vec qxx, qyy;
  double *qxx_p, *qyy_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qxx); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  Vec qzz;
  double *qzz_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &qzz); CHKERRXX(ierr);
#endif

  if(order>=1) { ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr); }

  for(int it=0; it<iterations; ++it)
  {
#ifdef P4_TO_P8
    ngbd->second_derivatives_central(q, qxx, qyy, qzz);
#else
    ngbd->second_derivatives_central(q, qxx, qyy);
#endif

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map=0; n_map<layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(mask_p[n] > -EPS)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
        dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
        dt /= 3;
#else
        dt /= 2;
#endif

        /* first order one sided derivatives */
        double qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                            : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                            : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                            : (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        /* second order derivatives */
        double qxx_m00 = qnnn.f_m00_linear(qxx_p);
        double qxx_p00 = qnnn.f_p00_linear(qxx_p);
        double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
        double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
        double qzz_00m = qnnn.f_00m_linear(qzz_p);
        double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

        /* minmod operation */
        qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
        else        qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
        else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

        tmp_p[n] = q_p[n] - dt*( nx[n]*qx + ny[n]*qy
                         #ifdef P4_TO_P8
                                 + nz[n]*qz
                         #endif
                                 ) + (order>=1 ? dt*qn_p[n]*region_p[n] : 0);
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map=0; n_map<local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(mask_p[n] > -EPS)
      {
        double dt = MIN(fabs(qnnn.d_m00) , fabs(qnnn.d_p00) );
        dt  =  MIN( dt, fabs(qnnn.d_0m0) , fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN( dt, fabs(qnnn.d_00m) , fabs(qnnn.d_00p) );
        dt /= 3;
#else
        dt /= 2;
#endif

        /* first order one sided derivatives */
        double qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                            : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                            : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                            : (qnnn.f_00p_linear(q_p) - q_p[n]) / qnnn.d_00p;
#endif

        /* second order derivatives */
        double qxx_m00 = qnnn.f_m00_linear(qxx_p);
        double qxx_p00 = qnnn.f_p00_linear(qxx_p);
        double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
        double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
        double qzz_00m = qnnn.f_00m_linear(qzz_p);
        double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

        /* minmod operation */
        qxx_m00 = qxx_p[n]*qxx_m00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00<0 ? 0 : (fabs(qxx_p[n])<fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0<0 ? 0 : (fabs(qyy_p[n])<fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p<0 ? 0 : (fabs(qzz_p[n])<fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
        else        qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
        else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

        tmp_p[n] = q_p[n] - dt*( nx[n]*qx + ny[n]*qy
                         #ifdef P4_TO_P8
                                 + nz[n]*qz
                         #endif
                                 ) + (order>=1 ? dt*qn_p[n]*region_p[n] : 0);
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* end update communication */
    ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif
    ierr = VecRestoreArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

    ierr = VecGhostGetLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(q  , &q_loc  ); CHKERRXX(ierr);
    ierr = VecCopy(tmp_loc, q_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(tmp, &tmp_loc); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(q  , &q_loc  ); CHKERRXX(ierr);
  }

  if(order>=1)
  {
    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecDestroy(qn); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(region, &region_p); CHKERRXX(ierr);

  ierr = VecDestroy(qxx); CHKERRXX(ierr);
  ierr = VecDestroy(qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecDestroy(qzz); CHKERRXX(ierr);
#endif

  ierr = VecDestroy(tmp); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_over_interface_TVD, phi, q, 0, 0); CHKERRXX(ierr);
}
