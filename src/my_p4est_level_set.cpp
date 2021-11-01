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
extern PetscLogEvent log_my_p4est_level_set_reinit_1_iter_1st_order;
extern PetscLogEvent log_my_p4est_level_set_reinit_1_iter_2nd_order;
extern PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_1_iter;
extern PetscLogEvent log_my_p4est_level_set_reinit_1st_order;
extern PetscLogEvent log_my_p4est_level_set_reinit_2nd_order;
extern PetscLogEvent log_my_p4est_level_set_reinit_1st_time_2nd_space;
extern PetscLogEvent log_my_p4est_level_set_reinit_2nd_time_1st_space;
extern PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_CF2;
extern PetscLogEvent log_my_p4est_level_set_advect_in_normal_direction_Vec;
extern PetscLogEvent log_my_p4est_level_set_geometric_extrapolation_over_interface;
extern PetscLogEvent log_my_p4est_level_set_extend_over_interface_TVD;
extern PetscLogEvent log_my_p4est_level_set_extend_from_interface;
extern PetscLogEvent log_my_p4est_level_set_extend_from_interface_TVD;
extern PetscLogEvent log_my_p4est_level_set_compute_derivatives;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

void my_p4est_level_set_t::compute_derivatives(Vec phi, Vec phi_xxyyzz[P4EST_DIM]) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_compute_derivatives, 0, 0, 0, 0); CHKERRXX(ierr);

  ngbd->second_derivatives_central(phi, phi_xxyyzz);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_compute_derivatives, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_level_set_t::reinitialize_one_iteration(double *phi_np1_p, const std::vector<p4est_locidx_t> &list_of_node_idx,
                                                      const double *phi_0_p, const double *phi_n_p,
                                                      const double &phi_0_limit_high_, const double &phi_0_limit_low_,
                                                      const double *phi_0_xxyyzz_p[P4EST_DIM], const double *phi_n_xxyyzz_p[P4EST_DIM]) const
{
  PetscErrorCode ierr;
  const bool second_order = phi_0_xxyyzz_p != NULL && phi_n_xxyyzz_p != NULL;
#ifdef CASL_LOG_EVENTS
  if(second_order)
    ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_1_iter_2nd_order, 0, 0, 0, 0);
  else
    ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_1_iter_1st_order, 0, 0, 0, 0);
#endif

  P4EST_ASSERT(phi_0_limit_high_ > phi_0_limit_low_);

  const double phi_0_limit_low  = (phi_0_limit_low_   < -largest_dbl_smaller_than_dbl_max ? -DBL_MAX : phi_0_limit_low_   - MAX(zero_distance_threshold, EPS*fabs(phi_0_limit_low_)));
  const double phi_0_limit_high = (phi_0_limit_high_  >  largest_dbl_smaller_than_dbl_max ?  DBL_MAX : phi_0_limit_high_  + MAX(zero_distance_threshold, EPS*fabs(phi_0_limit_high_)));

  const bool node_neighbors_are_initialized = ngbd->neighbors_are_initialized();
  quad_neighbor_nodes_of_node_t qnnn;
  const quad_neighbor_nodes_of_node_t *qnnn_p = (node_neighbors_are_initialized ? NULL : &qnnn); // we'll avoid data copy if possible!

  for(size_t k = 0; k < list_of_node_idx.size(); ++k)
  {
    const p4est_locidx_t& node_idx = list_of_node_idx[k];

    if(fabs(phi_0_p[node_idx]) < zero_distance_threshold)
      phi_np1_p[node_idx] = 0.0;
    else if(phi_0_limit_low < phi_0_p[node_idx] && phi_0_p[node_idx] < phi_0_limit_high)
    {
      if(node_neighbors_are_initialized)
        ngbd->get_neighbors(node_idx, qnnn_p);
      else
        ngbd->get_neighbors(node_idx, qnnn);

      const double *inputs[2] = {phi_0_p, phi_n_p};
      double outputs_000[2], outputs_m00[2], outputs_p00[2], outputs_0m0[2], outputs_0p0[2] ONLY3D(COMMA outputs_00m[2] COMMA outputs_00p[2]);
      qnnn_p->ngbd_with_quadratic_interpolation(inputs, outputs_000, outputs_m00, outputs_p00, outputs_0m0, outputs_0p0 ONLY3D(COMMA outputs_00m COMMA outputs_00p), 2);
      const double &phi_0_000 = outputs_000[0], &phi_0_m00 = outputs_m00[0], &phi_0_p00 = outputs_p00[0], &phi_0_0m0 = outputs_0m0[0], &phi_0_0p0 = outputs_0p0[0] ONLY3D(COMMA &phi_0_00m = outputs_00m[0] COMMA &phi_0_00p = outputs_00p[0]);
      double &phi_n_000 = outputs_000[1], &phi_n_m00 = outputs_m00[1], &phi_n_p00 = outputs_p00[1], &phi_n_0m0 = outputs_0m0[1], &phi_n_0p0 = outputs_0p0[1] ONLY3D(COMMA &phi_n_00m = outputs_00m[1] COMMA &phi_n_00p = outputs_00p[1]);

      double s_p00 = qnnn_p->d_p00; double s_m00 = qnnn_p->d_m00;
      double s_0p0 = qnnn_p->d_0p0; double s_0m0 = qnnn_p->d_0m0;
#ifdef P4_TO_P8
      double s_00p = qnnn_p->d_00p; double s_00m = qnnn_p->d_00m;
#endif

      //---------------------------------------------------------------------
      // Second Order derivatives - if provided (second-order)
      //---------------------------------------------------------------------
      double DIM(phi_n_xx_000 = NAN, phi_n_yy_000 = NAN, phi_n_zz_000 = NAN), DIM(phi_n_xx_m00 = NAN, phi_n_yy_0m0 = NAN, phi_n_zz_00m = NAN), DIM(phi_n_xx_p00 = NAN, phi_n_yy_0p0 = NAN, phi_n_zz_00p = NAN); // initialized to NAN so the compiler doesn't complain

      if(second_order)
      {
        phi_n_xx_000 = phi_n_xxyyzz_p[0][node_idx];
        phi_n_yy_000 = phi_n_xxyyzz_p[1][node_idx];
#ifdef P4_TO_P8
        phi_n_zz_000 = phi_n_xxyyzz_p[2][node_idx];
#endif
        phi_n_xx_m00 = qnnn_p->f_m00_linear(phi_n_xxyyzz_p[0]);
        phi_n_xx_p00 = qnnn_p->f_p00_linear(phi_n_xxyyzz_p[0]);
        phi_n_yy_0m0 = qnnn_p->f_0m0_linear(phi_n_xxyyzz_p[1]);
        phi_n_yy_0p0 = qnnn_p->f_0p0_linear(phi_n_xxyyzz_p[1]);
#ifdef P4_TO_P8
        phi_n_zz_00m = qnnn_p->f_00m_linear(phi_n_xxyyzz_p[2]);
        phi_n_zz_00p = qnnn_p->f_00p_linear(phi_n_xxyyzz_p[2]);
#endif
      }


      //---------------------------------------------------------------------
      // check if the node is near interface
      //---------------------------------------------------------------------
      if (phi_0_000*phi_0_m00 < 0.0 || phi_0_000*phi_0_p00 < 0.0 || phi_0_000*phi_0_0m0 < 0.0 || phi_0_000*phi_0_0p0 < 0.0 ONLY3D(|| phi_0_000*phi_0_00m < 0.0 || phi_0_000*phi_0_00p < 0.0))
      {
        double DIM(phi_0_xx_000 = NAN, phi_0_yy_000 = NAN, phi_0_zz_000 = NAN), DIM(phi_0_xx_m00 = NAN, phi_0_yy_0m0 = NAN, phi_0_zz_00m = NAN), DIM(phi_0_xx_p00 = NAN, phi_0_yy_0p0 = NAN, phi_0_zz_00p = NAN); // initialized to NAN so the compiler doesn't complain
        if(second_order)
        {
          phi_0_xx_000 = phi_0_xxyyzz_p[0][node_idx];
          phi_0_yy_000 = phi_0_xxyyzz_p[1][node_idx];
#ifdef P4_TO_P8
          phi_0_zz_000 = phi_0_xxyyzz_p[2][node_idx];
#endif
          phi_0_xx_m00 = qnnn_p->f_m00_linear(phi_0_xxyyzz_p[0]);
          phi_0_xx_p00 = qnnn_p->f_p00_linear(phi_0_xxyyzz_p[0]);
          phi_0_yy_0m0 = qnnn_p->f_0m0_linear(phi_0_xxyyzz_p[1]);
          phi_0_yy_0p0 = qnnn_p->f_0p0_linear(phi_0_xxyyzz_p[1]);
#ifdef P4_TO_P8
          phi_0_zz_00m = qnnn_p->f_00m_linear(phi_0_xxyyzz_p[2]);
          phi_0_zz_00p = qnnn_p->f_00p_linear(phi_0_xxyyzz_p[2]);
#endif
        }
        if(phi_0_000*phi_0_m00 < 0.0)
        {
          if(second_order)
            s_m00 = -interface_Location_With_Second_Order_Derivative(-s_m00, 0.0, phi_0_m00, phi_0_000, phi_0_xx_m00, phi_0_xx_000);
          else
            s_m00 = -interface_Location(-s_m00, 0.0, phi_0_m00, phi_0_000);
          phi_n_m00 = 0.0;
        }
        if(phi_0_000*phi_0_p00 < 0.0)
        {
          if(second_order)
            s_p00 = interface_Location_With_Second_Order_Derivative(0.0, s_p00, phi_0_000, phi_0_p00, phi_0_xx_000, phi_0_xx_p00);
          else
            s_p00 = interface_Location(0.0, s_p00, phi_0_000, phi_0_p00);
          phi_n_p00 = 0.0;
        }
        if(phi_0_000*phi_0_0m0 < 0.0)
        {
          if(second_order)
            s_0m0 = -interface_Location_With_Second_Order_Derivative(-s_0m0, 0.0, phi_0_0m0, phi_0_000, phi_0_yy_0m0, phi_0_yy_000);
          else
            s_0m0 = -interface_Location(-s_0m0, 0.0, phi_0_0m0, phi_0_000);
          phi_n_0m0 = 0.0;
        }
        if(phi_0_000*phi_0_0p0 < 0.0)
        {
          if(second_order)
            s_0p0 = interface_Location_With_Second_Order_Derivative(0.0, s_0p0, phi_0_000, phi_0_0p0, phi_0_yy_000, phi_0_yy_0p0);
          else
            s_0p0 = interface_Location(0.0, s_0p0, phi_0_000, phi_0_0p0);
          phi_n_0p0 = 0.0;
        }
#ifdef P4_TO_P8
        if(phi_0_000*phi_0_00m < 0.0)
        {
          if(second_order)
            s_00m = -interface_Location_With_Second_Order_Derivative(-s_00m, 0.0, phi_0_00m, phi_0_000, phi_0_zz_00m, phi_0_zz_000);
          else
            s_00m = -interface_Location(-s_00m, 0.0, phi_0_00m, phi_0_000);
          phi_n_00m = 0.0;
        }
        if(phi_0_000*phi_0_00p < 0.0)
        {
          if(second_order)
            s_00p = interface_Location_With_Second_Order_Derivative(0.0, s_00p, phi_0_000, phi_0_00p, phi_0_zz_000, phi_0_zz_00p);
          else
            s_00p = interface_Location(0.0, s_00p, phi_0_000, phi_0_00p);
          phi_n_00p = 0.0;
        }
#endif

        s_m00 = MAX(s_m00, zero_distance_threshold);
        s_p00 = MAX(s_p00, zero_distance_threshold);
        s_0m0 = MAX(s_0m0, zero_distance_threshold);
        s_0p0 = MAX(s_0p0, zero_distance_threshold);
#ifdef P4_TO_P8
        s_00m = MAX(s_00m, zero_distance_threshold);
        s_00p = MAX(s_00p, zero_distance_threshold);
#endif
      }

      //---------------------------------------------------------------------
      // Neumann boundary condition on the walls
      //---------------------------------------------------------------------
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, node_idx);

      if (is_node_xmWall(p4est, node)) { s_m00 = s_p00; phi_n_m00 = phi_n_p00; if(second_order) { phi_n_xx_000 = phi_n_xx_m00 = phi_n_xx_p00 = 0.0; }}
      if (is_node_xpWall(p4est, node)) { s_p00 = s_m00; phi_n_p00 = phi_n_m00; if(second_order) { phi_n_xx_000 = phi_n_xx_m00 = phi_n_xx_p00 = 0.0; }}
      if (is_node_ymWall(p4est, node)) { s_0m0 = s_0p0; phi_n_0m0 = phi_n_0p0; if(second_order) { phi_n_yy_000 = phi_n_yy_0m0 = phi_n_yy_0p0 = 0.0; }}
      if (is_node_ypWall(p4est, node)) { s_0p0 = s_0m0; phi_n_0p0 = phi_n_0m0; if(second_order) { phi_n_yy_000 = phi_n_yy_0m0 = phi_n_yy_0p0 = 0.0; }}
#ifdef P4_TO_P8
      if (is_node_zmWall(p4est, node)) { s_00m = s_00p; phi_n_00m = phi_n_00p; if(second_order) { phi_n_zz_000 = phi_n_zz_00m = phi_n_zz_00p = 0.0; }}
      if (is_node_zpWall(p4est, node)) { s_00p = s_00m; phi_n_00p = phi_n_00m; if(second_order) { phi_n_zz_000 = phi_n_zz_00m = phi_n_zz_00p = 0.0; }}
#endif

      //---------------------------------------------------------------------
      // First Order One-Sided Differecing
      //---------------------------------------------------------------------
      double phi_n_x_p00 = (phi_n_p00 - phi_n_000)/s_p00; double phi_n_x_m00 = (phi_n_000 - phi_n_m00)/s_m00;
      double phi_n_y_0p0 = (phi_n_0p0 - phi_n_000)/s_0p0; double phi_n_y_0m0 = (phi_n_000 - phi_n_0m0)/s_0m0;
#ifdef P4_TO_P8
      double phi_n_z_00p = (phi_n_00p - phi_n_000)/s_00p; double phi_n_z_00m = (phi_n_000 - phi_n_00m)/s_00m;
#endif
      if(second_order)
      {
        //---------------------------------------------------------------------
        // Second Order One-Sided Differencing
        //---------------------------------------------------------------------
        phi_n_xx_m00 = MINMOD(phi_n_xx_m00, phi_n_xx_000);  phi_n_x_m00 += 0.5*s_m00*phi_n_xx_m00;
        phi_n_xx_p00 = MINMOD(phi_n_xx_p00, phi_n_xx_000);  phi_n_x_p00 -= 0.5*s_p00*phi_n_xx_p00;
        phi_n_yy_0m0 = MINMOD(phi_n_yy_0m0, phi_n_yy_000);  phi_n_y_0m0 += 0.5*s_0m0*phi_n_yy_0m0;
        phi_n_yy_0p0 = MINMOD(phi_n_yy_0p0, phi_n_yy_000);  phi_n_y_0p0 -= 0.5*s_0p0*phi_n_yy_0p0;
#ifdef P4_TO_P8
        phi_n_zz_00m = MINMOD(phi_n_zz_00m, phi_n_zz_000);  phi_n_z_00m += 0.5*s_00m*phi_n_zz_00m;
        phi_n_zz_00p = MINMOD(phi_n_zz_00p, phi_n_zz_000);  phi_n_z_00p -= 0.5*s_00p*phi_n_zz_00p;
#endif
      }

      const double sgn = (phi_0_000 > 0.0 ? 1.0 : -1.0);

      //---------------------------------------------------------------------
      // Upwind Scheme
      //---------------------------------------------------------------------
      double dt = MIN(s_m00, s_p00);
      dt = MIN(dt, s_0m0, s_0p0);
#ifdef P4_TO_P8
      dt = MIN(dt, s_00m, s_00p);
#endif
      dt /= (double) P4EST_DIM;

      if(sgn*phi_n_x_p00 > 0.0) phi_n_x_p00 = 0.0;
      if(sgn*phi_n_x_m00 < 0.0) phi_n_x_m00 = 0.0;
      if(sgn*phi_n_y_0p0 > 0.0) phi_n_y_0p0 = 0.0;
      if(sgn*phi_n_y_0m0 < 0.0) phi_n_y_0m0 = 0.0;
#ifdef P4_TO_P8
      if(sgn*phi_n_z_00p > 0.0) phi_n_z_00p = 0.0;
      if(sgn*phi_n_z_00m < 0.0) phi_n_z_00m = 0.0;
#endif

      phi_np1_p[node_idx] = phi_n_000 - dt*sgn*(sqrt(SUMD(MAX(phi_n_x_p00*phi_n_x_p00, phi_n_x_m00*phi_n_x_m00), MAX(phi_n_y_0p0*phi_n_y_0p0, phi_n_y_0m0*phi_n_y_0m0), MAX(phi_n_z_00p*phi_n_z_00p, phi_n_z_00m*phi_n_z_00m))) - 1.);

      if(phi_0_000*phi_np1_p[node_idx] < 0.0)
        phi_np1_p[node_idx] *= -1.0; // do not allow for a change of sign, I guess

      ierr = PetscLogFlops(second_order ? 30 : 17); CHKERRXX(ierr);
    }
    /* else : outside the desired range for phi_0... nothing to do */
    else
      phi_np1_p[node_idx] = phi_0_p[node_idx];
  }
#ifdef CASL_LOG_EVENTS
  if(second_order)
    ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_1_iter_2nd_order, 0, 0, 0, 0);
  else
    ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_1_iter_1st_order, 0, 0, 0, 0);
#endif
}

void my_p4est_level_set_t::reinitialize_within_range_of_phi_0( Vec phi, const unsigned char order_space,
															   const unsigned char order_pseudotime,
                                                               const double &phi_0_limit_high,
                                                               const double &phi_0_limit_low,
                                                               const int &number_of_iterations,
															   const std::vector<p4est_locidx_t> *layer_nodes_p,
															   const std::vector<p4est_locidx_t> *local_nodes_p,
															   const std::vector<p4est_locidx_t> *masked_layer_nodes_p,
															   const std::vector<p4est_locidx_t> *masked_local_nodes_p ) const
{
  PetscErrorCode ierr;
  if(order_space == 0 || order_space > 2 || order_pseudotime == 0 || order_pseudotime > 2)
    throw std::invalid_argument("my_p4est_level_set_t::reinitialize_within_range_of_phi_0: valid orders of orders of accuracy for space and pseudotime are 1 and 2 only!");

  if(( layer_nodes_p ||  local_nodes_p ||  masked_layer_nodes_p ||  masked_local_nodes_p) &&
  	 (!layer_nodes_p || !local_nodes_p || !masked_layer_nodes_p || !masked_local_nodes_p))
  	throw std::invalid_argument("my_p4est_level_set_t::reinitialize_within_range_of_phi_0: not providing all custom layer and local node vectors!");
#ifdef CASL_LOG_EVENTS
  if(order_pseudotime == 1 && order_space == 1){
    ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_1st_order, phi, 0, 0, 0); CHKERRXX(ierr); }
  else if (order_pseudotime == 2 && order_space == 1) {
    ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_2nd_time_1st_space, phi, 0, 0, 0); CHKERRXX(ierr); }
  else if (order_pseudotime == 1 && order_space == 2) {
    ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_1st_time_2nd_space, phi, 0, 0, 0); CHKERRXX(ierr); }
  else {
    ierr = PetscLogEventBegin(log_my_p4est_level_set_reinit_2nd_order, phi, 0, 0, 0); CHKERRXX(ierr); }
#endif

  // we'll need phi_n
  double *phi_n_p;
  ierr = VecGetArray(phi, &phi_n_p); CHKERRXX(ierr);
  // we'll need a phi_np1, create it
  Vec phi_np1; double *phi_np1_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_np1); CHKERRXX(ierr);
  ierr = VecGetArray(phi_np1, &phi_np1_p);            CHKERRXX(ierr);
  // if second-order pseudo-time accuracy is required, we'll need a phi_np2 as well
  Vec phi_np2 = NULL; double *phi_np2_p = NULL;
  if(order_pseudotime > 1) {
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_np2); CHKERRXX(ierr);
    ierr = VecGetArray(phi_np2, &phi_np2_p);            CHKERRXX(ierr);
  }
  // we need to copy the given phi into phi_0 for interface anchoring:
  double *phi_0_p = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
  memcpy(phi_0_p, phi_n_p, nodes->indep_nodes.elem_count*sizeof(double));

  // if second-order space accuracy is desired, we'll need the second derivatives of phi_0 and the 'current_phi' as well
  Vec *phi_0_xxyyzz       = NULL; const double **phi_0_xxyyzz_read_p = NULL;
  Vec *current_phi_xxyyzz = NULL; const double **current_phi_xxyyzz_read_p = NULL;
  if(order_space > 1)
  {
    phi_0_xxyyzz = new Vec[P4EST_DIM];        phi_0_xxyyzz_read_p       = new const double*[P4EST_DIM];
    current_phi_xxyyzz = new Vec[P4EST_DIM];  current_phi_xxyyzz_read_p = new const double*[P4EST_DIM];
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecCreateGhostNodes(p4est, nodes, &phi_0_xxyyzz[dim]);       CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &current_phi_xxyyzz[dim]); CHKERRXX(ierr);
    }
    compute_derivatives(phi, phi_0_xxyyzz);
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecGetArrayRead(phi_0_xxyyzz[dim], &phi_0_xxyyzz_read_p[dim]); CHKERRXX(ierr);
    }
  }

  for(int i = 0; i < number_of_iterations; i++)
  {
    /***** Step 1: phi -> phi_np1 *****/
    if(order_space > 1)
    {
      /* compute derivatives */
      compute_derivatives(phi, current_phi_xxyyzz);
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecGetArrayRead(current_phi_xxyyzz[dim], &current_phi_xxyyzz_read_p[dim]); CHKERRXX(ierr);
      }
    }

#ifdef IPM_LOG_EVENTS
    if(order_pseudotime == 1 && order_space == 1)
      IPMLogRegionBegin("reinit_1st_1st");
    else if (order_pseudotime == 2 && order_space == 1)
      IPMLogRegionBegin("reinit_2nd_1st");
    else if (order_pseudotime == 1 && order_space == 2)
      IPMLogRegionBegin("reinit_1st_2nd");
    else
      IPMLogRegionBegin("reinit_2nd_2nd");
#endif

    /* 1) Process layer nodes */
    reinitialize_one_iteration(phi_np1_p, (layer_nodes_p? *layer_nodes_p : ngbd->layer_nodes),
    						   phi_0_p, phi_n_p, phi_0_limit_high, phi_0_limit_low,
                               phi_0_xxyyzz_read_p, current_phi_xxyyzz_read_p);
    if(layer_nodes_p)
    {
      for(const auto& n : *masked_layer_nodes_p)	// Copy values for masked layer nodes.
        phi_np1_p[n] = phi_n_p[n];
    }

    /* 2) Begin update process for phi_np1 */
    ierr = VecGhostUpdateBegin(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* 3) Process local nodes */
    reinitialize_one_iteration(phi_np1_p, (local_nodes_p? *local_nodes_p : ngbd->local_nodes),
							   phi_0_p, phi_n_p, phi_0_limit_high, phi_0_limit_low,
                               phi_0_xxyyzz_read_p, current_phi_xxyyzz_read_p);

    if(local_nodes_p)
    {
      for(const auto& n : *masked_local_nodes_p)	// Copy values for masked local nodes.
        phi_np1_p[n] = phi_n_p[n];
    }

    /* 4) End update process for phi_np1 */
    ierr = VecGhostUpdateEnd(phi_np1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    if(order_space > 1)
    {
      for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
        ierr = VecRestoreArrayRead(current_phi_xxyyzz[dim], &current_phi_xxyyzz_read_p[dim]); CHKERRXX(ierr);
      }
    }

#ifdef IPM_LOG_EVENTS
    if(order_pseudotime == 1 && order_space == 1)
      IPMLogRegionEnd("reinit_1st_1st");
    else if (order_pseudotime == 2 && order_space == 1)
      IPMLogRegionEnd("reinit_2nd_1st");
    else if (order_pseudotime == 1 && order_space == 2)
      IPMLogRegionEnd("reinit_1st_2nd");
    else
      IPMLogRegionEnd("reinit_2nd_2nd");
#endif

    if(order_pseudotime > 1)
    {
      if(order_space > 1)
      {
        /* recompute derivatives */
        compute_derivatives(phi_np1, current_phi_xxyyzz);
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
          ierr = VecGetArrayRead(current_phi_xxyyzz[dim], &current_phi_xxyyzz_read_p[dim]); CHKERRXX(ierr);
        }
      }

      /***** Step 2 of RK2: phi_np1 -> phi_np2 *****/
#ifdef IPM_LOG_EVENTS
      if(order_space == 1)
        IPMLogRegionBegin("reinit_2nd_1st");
      else
        IPMLogRegionBegin("reinit_2nd_2nd");
#endif
      /* 1) Process layer nodes */
      reinitialize_one_iteration(phi_np2_p, (layer_nodes_p? *layer_nodes_p : ngbd->layer_nodes),
								 phi_0_p, phi_np1_p, phi_0_limit_high, phi_0_limit_low,
                                 phi_0_xxyyzz_read_p, current_phi_xxyyzz_read_p);

      if(layer_nodes_p)
      {
        for(const auto& n : *masked_layer_nodes_p)	// Copy values for masked layer nodes.
          phi_np2_p[n] = phi_np1_p[n];
      }

      /* 2) Begin update process for phi_np2 */
      ierr = VecGhostUpdateBegin(phi_np2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* 3) Process local nodes */
      reinitialize_one_iteration(phi_np2_p, (local_nodes_p? *local_nodes_p : ngbd->local_nodes),
								 phi_0_p, phi_np1_p, phi_0_limit_high, phi_0_limit_low,
                                 phi_0_xxyyzz_read_p, current_phi_xxyyzz_read_p);

      if(local_nodes_p)
      {
        for(const auto& n : *masked_local_nodes_p)	// Copy values for masked local nodes.
          phi_np2_p[n] = phi_np1_p[n];
      }

      /* 4) End update process for phi_np2 */
      ierr = VecGhostUpdateEnd(phi_np2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef IPM_LOG_EVENTS
      if(order_space == 1)
        IPMLogRegionEnd("reinit_2nd_1st");
      else
        IPMLogRegionEnd("reinit_2nd_2nd");
#endif
      if(order_space > 1)
      {
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
          ierr = VecRestoreArrayRead(current_phi_xxyyzz[dim], &current_phi_xxyyzz_read_p[dim]); CHKERRXX(ierr);
        }
      }

      if (show_convergence)
      {
        double dxyz[P4EST_DIM];
        ::dxyz_min(p4est, dxyz);

        double change = 0.0;
        for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
          if (fabs(phi_n_p[n]) < show_convergence_band*dxyz[0])
          {
            double change_loc = 0.5*fabs(phi_np2_p[n] - phi_n_p[n]);
            if (change_loc > change) change = change_loc;
          }

        int mpiret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

        ierr = PetscPrintf(p4est->mpicomm, "Reinitializing, iteration: %d, error: %e\n", i, change); CHKERRXX(ierr);
      }

      /* update phi */
      for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
        phi_n_p[n] = .5*(phi_n_p[n] + phi_np2_p[n]);
    }
    else
    {
      if (show_convergence)
      {
        double dxyz[P4EST_DIM];
        ::dxyz_min(p4est, dxyz);

        double change = 0.0;
        for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
          if (fabs(phi_n_p[n]) < show_convergence_band*dxyz[0])
          {
            double change_loc = fabs(phi_np1_p[n] - phi_n_p[n]);
            if (change_loc > change) change = change_loc;
          }

        int mpiret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

        ierr = PetscPrintf(p4est->mpicomm, "Reinitializing, iteration: %d, error: %e\n", i, change); CHKERRXX(ierr);
      }

      /* update phi */
      memcpy(phi_n_p, phi_np1_p, nodes->indep_nodes.elem_count*sizeof(double));

    }
  }

  /* restore arrays and destroy locally created petsc vectors */
  ierr = VecRestoreArray(phi,     &phi_n_p);      CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_np1, &phi_np1_p);    CHKERRXX(ierr);
  ierr = VecDestroy(phi_np1);                     CHKERRXX(ierr);
  if(order_pseudotime > 1){
    ierr = VecRestoreArray(phi_np2, &phi_np2_p);  CHKERRXX(ierr);
    ierr = VecDestroy(phi_np2);                   CHKERRXX(ierr);
  }
  if(order_space > 1)
  {
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArrayRead(phi_0_xxyyzz[dim], &phi_0_xxyyzz_read_p[dim]); CHKERRXX(ierr);
      ierr = VecDestroy(phi_0_xxyyzz[dim]);       CHKERRXX(ierr);
      ierr = VecDestroy(current_phi_xxyyzz[dim]); CHKERRXX(ierr);
    }
  }
  free(phi_0_p);

#ifdef CASL_LOG_EVENTS
  if(order_pseudotime == 1 && order_space == 1){
    ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_1st_order, phi, 0, 0, 0); CHKERRXX(ierr); }
  else if (order_pseudotime == 2 && order_space == 1) {
    ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_2nd_time_1st_space, phi, 0, 0, 0); CHKERRXX(ierr); }
  else if (order_pseudotime == 1 && order_space == 2) {
    ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_1st_time_2nd_space, phi, 0, 0, 0); CHKERRXX(ierr); }
  else {
    ierr = PetscLogEventEnd(log_my_p4est_level_set_reinit_2nd_order, phi, 0, 0, 0); CHKERRXX(ierr); }
#endif
  return;
}

void my_p4est_level_set_t::advect_in_normal_direction_one_iteration(const std::vector<p4est_locidx_t> &list_of_node_idx, const double *vn, const double &dt,
                                                                    const double *phi_n_xxyyzz_p[P4EST_DIM], const double *phi_n_p, double *phi_np1_p) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_advect_in_normal_direction_1_iter, 0, 0, 0, 0);

  const bool node_neighbors_are_initialized = ngbd->neighbors_are_initialized();
  quad_neighbor_nodes_of_node_t qnnn;
  const quad_neighbor_nodes_of_node_t *qnnn_p = (node_neighbors_are_initialized ? NULL : &qnnn); // we'll avoid data copy if possible!

  for(size_t k = 0; k < list_of_node_idx.size(); ++k)
  {
    const p4est_locidx_t & node_idx = list_of_node_idx[k];

    if(node_neighbors_are_initialized)
      ngbd->get_neighbors(node_idx, qnnn_p);
    else
      ngbd->get_neighbors(node_idx, qnnn);

    double phi_n_000, phi_n_m00, phi_n_p00, phi_n_0m0, phi_n_0p0 ONLY3D(COMMA phi_n_00m COMMA phi_n_00p);
    qnnn_p->ngbd_with_quadratic_interpolation(phi_n_p, phi_n_000, phi_n_m00, phi_n_p00, phi_n_0m0, phi_n_0p0 ONLY3D(COMMA phi_n_00m COMMA phi_n_00p));
    double s_p00 = qnnn_p->d_p00; double s_m00 = qnnn_p->d_m00;
    double s_0p0 = qnnn_p->d_0p0; double s_0m0 = qnnn_p->d_0m0;
#ifdef P4_TO_P8
    double s_00p = qnnn_p->d_00p; double s_00m = qnnn_p->d_00m;
#endif

    //---------------------------------------------------------------------
    // Second Order derivatives
    //---------------------------------------------------------------------
    double phi_n_xx_000 = phi_n_xxyyzz_p[0][node_idx];
    double phi_n_yy_000 = phi_n_xxyyzz_p[1][node_idx];
#ifdef P4_TO_P8
    double phi_n_zz_000 = phi_n_xxyyzz_p[2][node_idx];
#endif
    double phi_n_xx_m00 = qnnn_p->f_m00_linear(phi_n_xxyyzz_p[0]);
    double phi_n_xx_p00 = qnnn_p->f_p00_linear(phi_n_xxyyzz_p[0]);
    double phi_n_yy_0m0 = qnnn_p->f_0m0_linear(phi_n_xxyyzz_p[1]);
    double phi_n_yy_0p0 = qnnn_p->f_0p0_linear(phi_n_xxyyzz_p[1]);
#ifdef P4_TO_P8
    double phi_n_zz_00m = qnnn_p->f_00m_linear(phi_n_xxyyzz_p[2]);
    double phi_n_zz_00p = qnnn_p->f_00p_linear(phi_n_xxyyzz_p[2]);
#endif

    //---------------------------------------------------------------------
    // Neumann boundary condition on the walls
    //---------------------------------------------------------------------
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, node_idx);

    if (is_node_xmWall(p4est, node)) { s_m00 = s_p00; phi_n_m00 = phi_n_p00; phi_n_xx_000 = phi_n_xx_m00 = phi_n_xx_p00 = 0.0; }
    if (is_node_xpWall(p4est, node)) { s_p00 = s_m00; phi_n_p00 = phi_n_m00; phi_n_xx_000 = phi_n_xx_m00 = phi_n_xx_p00 = 0.0; }
    if (is_node_ymWall(p4est, node)) { s_0m0 = s_0p0; phi_n_0m0 = phi_n_0p0; phi_n_yy_000 = phi_n_yy_0m0 = phi_n_yy_0p0 = 0.0; }
    if (is_node_ypWall(p4est, node)) { s_0p0 = s_0m0; phi_n_0p0 = phi_n_0m0; phi_n_yy_000 = phi_n_yy_0m0 = phi_n_yy_0p0 = 0.0; }
#ifdef P4_TO_P8
    if (is_node_zmWall(p4est, node)) { s_00m = s_00p; phi_n_00m = phi_n_00p; phi_n_zz_000 = phi_n_zz_00m = phi_n_zz_00p = 0.0; }
    if (is_node_zpWall(p4est, node)) { s_00p = s_00m; phi_n_00p = phi_n_00m; phi_n_zz_000 = phi_n_zz_00m = phi_n_zz_00p = 0.0; }
#endif

    //---------------------------------------------------------------------
    // First Order One-Sided Differecing
    //---------------------------------------------------------------------
    double phi_n_x_p00 = (phi_n_p00 - phi_n_000)/s_p00; double phi_n_x_m00 = (phi_n_000 - phi_n_m00)/s_m00;
    double phi_n_y_0p0 = (phi_n_0p0 - phi_n_000)/s_0p0; double phi_n_y_0m0 = (phi_n_000 - phi_n_0m0)/s_0m0;
#ifdef P4_TO_P8
    double phi_n_z_00p = (phi_n_00p - phi_n_000)/s_00p; double phi_n_z_00m = (phi_n_000 - phi_n_00m)/s_00m;
#endif

    //---------------------------------------------------------------------
    // Second Order One-Sided Differencing
    //---------------------------------------------------------------------
    phi_n_xx_m00 = MINMOD(phi_n_xx_m00, phi_n_xx_000);   phi_n_x_m00 += 0.5*s_m00*phi_n_xx_m00;
    phi_n_xx_p00 = MINMOD(phi_n_xx_p00, phi_n_xx_000);   phi_n_x_p00 -= 0.5*s_p00*phi_n_xx_p00;
    phi_n_yy_0m0 = MINMOD(phi_n_yy_0m0, phi_n_yy_000);   phi_n_y_0m0 += 0.5*s_0m0*phi_n_yy_0m0;
    phi_n_yy_0p0 = MINMOD(phi_n_yy_0p0, phi_n_yy_000);   phi_n_y_0p0 -= 0.5*s_0p0*phi_n_yy_0p0;
#ifdef P4_TO_P8
    phi_n_zz_00m = MINMOD(phi_n_zz_00m, phi_n_zz_000);   phi_n_z_00m += 0.5*s_00m*phi_n_zz_00m;
    phi_n_zz_00p = MINMOD(phi_n_zz_00p, phi_n_zz_000);   phi_n_z_00p -= 0.5*s_00p*phi_n_zz_00p;
#endif

    const double sgn = (vn[node_idx] > 0.0 ? +1.0 : -1.0);

    if(sgn*phi_n_x_p00 > 0.0) phi_n_x_p00 = 0.0;
    if(sgn*phi_n_x_m00 < 0.0) phi_n_x_m00 = 0.0;
    if(sgn*phi_n_y_0p0 > 0.0) phi_n_y_0p0 = 0.0;
    if(sgn*phi_n_y_0m0 < 0.0) phi_n_y_0m0 = 0.0;
#ifdef P4_TO_P8
    if(sgn*phi_n_z_00p > 0.0) phi_n_z_00p = 0.0;
    if(sgn*phi_n_z_00m < 0.0) phi_n_z_00m = 0.0;
#endif

    phi_np1_p[node_idx] = phi_n_000 - dt*vn[node_idx]*(sqrt(SUMD(phi_n_x_p00*phi_n_x_p00 + phi_n_x_m00*phi_n_x_m00, phi_n_y_0p0*phi_n_y_0p0 + phi_n_y_0m0*phi_n_y_0m0, phi_n_z_00p*phi_n_z_00p + phi_n_z_00m*phi_n_z_00m)));

    ierr = PetscLogFlops(30); CHKERRXX(ierr);
  }
  ierr = PetscLogEventEnd(log_my_p4est_level_set_advect_in_normal_direction_1_iter, 0, 0, 0, 0);
}

void my_p4est_level_set_t::advect_in_normal_direction(Vec phi, const double &dt, DIM(Vec phi_xx, Vec phi_yy, Vec phi_zz),
                                                      const double *node_sampled_vn_p, const double *node_sampled_vnp1_p) const
{
  PetscErrorCode ierr;
  Vec current_phi_xxyyzz[P4EST_DIM] = {DIM(phi_xx, phi_yy, phi_zz)};
  bool local_derivatives = false;
  if (ANDD(current_phi_xxyyzz[0] == NULL, current_phi_xxyyzz[1] == NULL, current_phi_xxyyzz[2] == NULL))
  {
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecCreateGhostNodes(p4est, nodes, &current_phi_xxyyzz[dim]); CHKERRXX(ierr);
    }
    compute_derivatives(phi, current_phi_xxyyzz);
    local_derivatives = true;
  }

  double *current_phi_p;
  const double *current_phi_xxyyzz_p[P4EST_DIM];
  ierr = VecGetArray(phi, &current_phi_p); CHKERRXX(ierr);
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArrayRead(current_phi_xxyyzz[dim], &current_phi_xxyyzz_p[dim]); CHKERRXX(ierr);
  }

  Vec phi_n, phi_np2;
  double *phi_n_p, *phi_np2_p;
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_n); CHKERRXX(ierr);
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_np2); CHKERRXX(ierr);

  ierr = VecGetArray(phi_n, &phi_n_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_np2, &phi_np2_p); CHKERRXX(ierr);

  memcpy(phi_n_p, current_phi_p, sizeof(double) * nodes->indep_nodes.elem_count);
  // layer nodes
  advect_in_normal_direction_one_iteration(ngbd->layer_nodes, node_sampled_vn_p, dt, current_phi_xxyyzz_p, phi_n_p, current_phi_p);
  ierr = VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  // local nodes
  advect_in_normal_direction_one_iteration(ngbd->local_nodes, node_sampled_vn_p, dt, current_phi_xxyyzz_p, phi_n_p, current_phi_p);
  ierr = VecGhostUpdateEnd  (phi, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* now phi(Lnp1, Bnp1, Gnp1) */
  compute_derivatives(phi, current_phi_xxyyzz);

  // layer nodes
  advect_in_normal_direction_one_iteration(ngbd->layer_nodes, (node_sampled_vnp1_p != NULL ? node_sampled_vnp1_p : node_sampled_vn_p),
                                           dt, current_phi_xxyyzz_p, current_phi_p, phi_np2_p);
  ierr = VecGhostUpdateBegin(phi_np2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // local nodes
  advect_in_normal_direction_one_iteration(ngbd->local_nodes, (node_sampled_vnp1_p != NULL ? node_sampled_vnp1_p : node_sampled_vn_p),
                                           dt, current_phi_xxyyzz_p, current_phi_p, phi_np2_p);
  ierr = VecGhostUpdateEnd  (phi_np2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
    current_phi_p[n] = 0.5*(phi_n_p[n] + phi_np2_p[n]);

  /* restore arrays */
  ierr = VecRestoreArray(phi,     &current_phi_p);  CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_n,   &phi_n_p);        CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_np2, &phi_np2_p);      CHKERRXX(ierr);
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecRestoreArrayRead(current_phi_xxyyzz[dim], &current_phi_xxyyzz_p[dim]); CHKERRXX(ierr);
    if (local_derivatives){
      ierr = VecDestroy(current_phi_xxyyzz[dim]);   CHKERRXX(ierr);
    }
  }
  ierr = VecDestroy(phi_n);   CHKERRXX(ierr);
  ierr = VecDestroy(phi_np2); CHKERRXX(ierr);

  ierr = PetscLogFlops(nodes->num_owned_indeps * P4EST_DIM); CHKERRXX(ierr);

  return;
}

void my_p4est_level_set_t::perturb_level_set_function(Vec phi, const double &epsilon) const
{
  PetscErrorCode ierr;
  double *phi_ptr;

  ierr = VecGetArray(phi, &phi_ptr); CHKERRXX(ierr);

  for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
    if(fabs(phi_ptr[n]) < epsilon)
    {
      if(phi_ptr[n] > 0) phi_ptr[n] =  epsilon;
      else               phi_ptr[n] = -epsilon;
    }

  ierr = VecRestoreArray(phi, &phi_ptr); CHKERRXX(ierr);
}

double my_p4est_level_set_t::advect_in_normal_direction(const CF_DIM& vn, Vec phi, DIM(Vec phi_xx, Vec phi_yy, Vec phi_zz)) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_advect_in_normal_direction_CF2, 0, 0, 0, 0); CHKERRXX(ierr);

  /* compute dt based on CFL condition */
  double dt = DBL_MAX;
  std::vector<double> vn_vec(nodes->indep_nodes.elem_count);
  const bool node_neihbors_are_initialized = ngbd->neighbors_are_initialized();
  quad_neighbor_nodes_of_node_t qnnn;
  const quad_neighbor_nodes_of_node_t *qnnn_p = (node_neihbors_are_initialized ? NULL : &qnnn); // avoid data copy if possible
  for (p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n){
    if(node_neihbors_are_initialized)
      ngbd->get_neighbors(n, qnnn_p);
    else
      ngbd->get_neighbors(n, qnnn);

    double xyzn[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyzn);

    double s_p00 = fabs(qnnn_p->d_p00); double s_m00 = fabs(qnnn_p->d_m00);
    double s_0p0 = fabs(qnnn_p->d_0p0); double s_0m0 = fabs(qnnn_p->d_0m0);
#ifdef P4_TO_P8
    double s_00p = fabs(qnnn_p->d_00p); double s_00m = fabs(qnnn_p->d_00m);
#endif
    double s_min = MIN(DIM(MIN(s_p00, s_m00), MIN(s_0p0, s_0m0), MIN(s_00p, s_00m)));

    /* choose CFL = 0.8 ... just for fun! */
    vn_vec[n] = vn(xyzn);
    dt = MIN(dt, 0.8*fabs(s_min/vn_vec[n]));
  }
  int mpiret = MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN, p4est->mpicomm); SC_CHECK_MPI(mpiret);

  advect_in_normal_direction(phi, dt, DIM(phi_xx, phi_yy, phi_zz), &vn_vec[0]);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_advect_in_normal_direction_CF2, 0, 0, 0, 0); CHKERRXX(ierr);

  return dt;
}

double my_p4est_level_set_t::advect_in_normal_direction(const Vec vn, Vec phi, const double &dt, DIM(Vec phi_xx, Vec phi_yy, Vec phi_zz)) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_advect_in_normal_direction_Vec, 0, 0, 0, 0);

  const double *vn_read_p;
  ierr = VecGetArrayRead(vn, &vn_read_p); CHKERRXX(ierr);
  advect_in_normal_direction(phi, dt, DIM(phi_xx, phi_yy, phi_zz), vn_read_p);
  ierr = VecRestoreArrayRead(vn, &vn_read_p); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_advect_in_normal_direction_Vec, 0, 0, 0, 0);

  return dt;
}

double my_p4est_level_set_t::advect_in_normal_direction(const Vec vn, const Vec vn_np1, Vec phi, const double &dt, DIM(Vec phi_xx, Vec phi_yy, Vec phi_zz)) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_advect_in_normal_direction_Vec, 0, 0, 0, 0);

  const double *vn_read_p, *vnp1_read_p;
  ierr = VecGetArrayRead(vn, &vn_read_p); CHKERRXX(ierr);
  ierr = VecGetArrayRead(vn_np1, &vnp1_read_p); CHKERRXX(ierr);
  advect_in_normal_direction(phi, dt, DIM(phi_xx, phi_yy, phi_zz), vn_read_p, vnp1_read_p);
  ierr = VecRestoreArrayRead(vn_np1, &vnp1_read_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(vn, &vn_read_p); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_advect_in_normal_direction_Vec, 0, 0, 0, 0);

  return dt;
}

void my_p4est_level_set_t::geometric_extrapolation_over_interface(Vec phi, Vec q, const BoundaryConditionsDIM *bc, const unsigned char &degree, const unsigned int &band_to_extend) const
{
#ifdef CASL_THROWS
  if(bc->interfaceType() == NOINTERFACE)
    throw std::invalid_argument("my_p4est_level_set_t::geometric_extrapolation_across_interface: no interface defined in the boundary condition ... needs to be dirichlet, neumann or mixed.");
  if(degree > 2)
    throw std::invalid_argument("my_p4est_level_set_t::geometric_extrapolation_across_interface: the degree of the extrapolant polynomial must be less than or equal to 2.");
#endif
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_geometric_extrapolation_over_interface, phi, q, 0, 0); CHKERRXX(ierr);

  // get the (max) number of points to sample in negative domain for every node where extrapolation is required
  const unsigned char nsamples = number_of_samples_across_the_interface_for_geometric_extrapolation(degree, bc->interfaceType());
  P4EST_ASSERT(nsamples <= 2);

  // find the smallest diag and set distances away from the interface for sampling values of q
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  const double diag_min = sqrt(SUMD(SQR(tree_dimensions[0]/pow(2.,(double) data->max_lvl)), SQR(tree_dimensions[1]/pow(2.,(double) data->max_lvl)), SQR(tree_dimensions[2]/pow(2.,(double) data->max_lvl))));
  // if bc is not given, we use values in negative domain only
  // the methodology is to use a DIRICHLET-like condition on the -sampling_distances[0] level of the levelset
  const double relative_zero_level    = (bc == NULL ? -2.0*diag_min : 0.0);           // signed distance from the 0-level in the normal direction
  const double relative_phi_level[2]  = {(bc == NULL ? -1.0*diag_min : -2.0*diag_min), (bc == NULL ? -2.0*diag_min : -3.0*diag_min)};

  // we'll need local node neighbors to calculate grad_phi
  const bool node_neighbors_are_initialized = ngbd->neighbors_are_initialized();
  quad_neighbor_nodes_of_node_t qnnn;
  const quad_neighbor_nodes_of_node_t *qnnn_p = (node_neighbors_are_initialized ? NULL : &qnnn); // avoid data copy if node neighbors are initialized

  // get ready
  const double *phi_p;
  ierr = VecGetArrayRead(phi, &phi_p); CHKERRXX(ierr);
  my_p4est_interpolation_nodes_t *interp_bc = NULL;
  if(bc != NULL)
    interp_bc = new my_p4est_interpolation_nodes_t(ngbd);
  my_p4est_interpolation_nodes_t interp_q(ngbd);
  interp_q.set_input(q, quadratic_non_oscillatory);
  std::map<p4est_locidx_t, data_for_geometric_extapolation> node_data_for_extrapolation; node_data_for_extrapolation.clear();
  /* now buffer the interpolation points */
  for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    if(phi_p[n] > 0.0 && phi_p[n] < band_to_extend*diag_min)
    {
      if(node_neighbors_are_initialized)
        ngbd->get_neighbors(n, qnnn_p);
      else
        ngbd->get_neighbors(n, qnnn);

      double xyz_node[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz_node);

      double grad_phi[P4EST_DIM];
      qnnn_p->gradient(phi_p, grad_phi);
      add_dof_to_extrapolation_map(node_data_for_extrapolation, n, xyz_node, phi_p[n] - relative_zero_level, grad_phi, nsamples,
                                   relative_phi_level, &interp_q, interp_bc);
    }

  std::vector<double> qsamples;
  std::vector<bc_sample> *calculated_bc_samples = NULL;
  qsamples.resize((nsamples + (bc == NULL))*node_data_for_extrapolation.size());
  interp_q.interpolate(qsamples.data());
  if(bc != NULL)
  {
    calculated_bc_samples = new std::vector<bc_sample>(node_data_for_extrapolation.size());
    interp_bc->evaluate_interface_bc(*bc, calculated_bc_samples->data());
  }

  /* now compute the extrapolated values */
  double *q_p;
  ierr = VecGetArray(q, &q_p); CHKERRXX(ierr);
  std::map<p4est_locidx_t, data_for_geometric_extapolation>::iterator it;
  for (size_t k = 0; k < ngbd->get_layer_size(); ++k) {
    p4est_locidx_t node_idx = ngbd->get_layer_node(k);
    it = node_data_for_extrapolation.find(node_idx);
    if(it != node_data_for_extrapolation.end())
      q_p[node_idx] = build_extrapolation_data_and_compute_geometric_extrapolation(it->second, degree, nsamples, qsamples, calculated_bc_samples);
  }
  ierr = VecGhostUpdateBegin(q, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  for (size_t k = 0; k < ngbd->get_local_size(); ++k) {
    p4est_locidx_t node_idx = ngbd->get_local_node(k);
    it = node_data_for_extrapolation.find(node_idx);
    if(it != node_data_for_extrapolation.end())
      q_p[node_idx] = build_extrapolation_data_and_compute_geometric_extrapolation(it->second, degree, nsamples, qsamples, calculated_bc_samples);
  }
  ierr = VecGhostUpdateEnd(q, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  if(bc!= NULL)
  {
    delete interp_bc;
    delete calculated_bc_samples;
  }
  ierr = PetscLogEventEnd(log_my_p4est_level_set_geometric_extrapolation_over_interface, phi, q, 0, 0); CHKERRXX(ierr);
}

void my_p4est_level_set_t::extend_from_interface_to_whole_domain(Vec phi, Vec q, Vec q_extended, const unsigned int &band_to_extend) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_from_interface, phi, q, q_extended, 0); CHKERRXX(ierr);

  // find the smallest diag and set distances away from the interface for sampling values of q
  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  const double diag_min = sqrt(SUMD(SQR(tree_dimensions[0]/pow(2.,(double) data->max_lvl)), SQR(tree_dimensions[1]/pow(2.,(double) data->max_lvl)), SQR(tree_dimensions[2]/pow(2.,(double) data->max_lvl))));

  my_p4est_interpolation_nodes_t interp(ngbd);

  double *q_extended_p;
  const double *phi_p;
  ierr = VecGetArray(q_extended, &q_extended_p);  CHKERRXX(ierr);
  ierr = VecGetArrayRead(phi, &phi_p);            CHKERRXX(ierr);

  // we'll need local node neighbors to calculate grad_phi
  const bool node_neighbors_are_initialized = ngbd->neighbors_are_initialized();
  quad_neighbor_nodes_of_node_t qnnn;
  const quad_neighbor_nodes_of_node_t *qnnn_p = (node_neighbors_are_initialized ? NULL : &qnnn); // avoid data copy if node neighbors are initialized

  for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
  {
    q_extended_p[n] = 0.0; // initialize value in case we don't go through the following conditions, to avoid NANs
    if(fabs(phi_p[n]) < band_to_extend*diag_min)
    {
      if(node_neighbors_are_initialized)
        ngbd->get_neighbors(n, qnnn_p);
      else
        ngbd->get_neighbors(n, qnnn);

      double normal[P4EST_DIM];
      qnnn_p->gradient(phi_p, normal);
      const double mag_normal = sqrt(SUMD(SQR(normal[0]), SQR(normal[1]), SQR(normal[2])));
      if(mag_normal > EPS)
      {
        double xyz_i[P4EST_DIM];
        node_xyz_fr_n(n, p4est, nodes, xyz_i);
        for (unsigned char dim = 0; dim < P4EST_DIM; ++dim)
          xyz_i[dim] -= normal[dim]*phi_p[n]/mag_normal; // we want to interpolate the _interface_ value, don't we? This projection was missing until 04/08/2020 [Raphael]
        interp.add_point(n, xyz_i);
      }
      ierr = PetscLogFlops(14); CHKERRXX(ierr);
    }
  }

  interp.set_input(q, quadratic);
  interp.interpolate(q_extended_p);

  ierr = VecRestoreArrayRead(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(q_extended, &q_extended_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(q_extended, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (q_extended, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_from_interface, phi, q, q_extended, 0); CHKERRXX(ierr);
}


void my_p4est_level_set_t::extend_Over_Interface_TVD(Vec phi, Vec q, int iterations, int order, double band_use, double band_extend,
                                                     Vec normal[], Vec mask, boundary_conditions_t *bc,
                                                     bool use_nonzero_guess, Vec q_n, Vec q_nn) const
{
#ifdef CASL_THROWS
  if(order != 0 && order != 1 && order != 2) throw std::invalid_argument("[CASL_ERROR]: my_p4est_level_set_t->extend_Over_Interface_TVD: order must be 0, 1 or 2.");
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
  // band_use is assumed to be negative but user can specify a positive value
  if (band_use > 0) band_use = -band_use;

  Vec b_qn_well_defined;  double *b_qn_well_defined_p;
  Vec b_qnn_well_defined; double *b_qnn_well_defined_p;

  double *q_p;

  Vec qn;  double *qn_p;
  Vec qnn; double *qnn_p;

  Vec qx; double *qx_p;
  Vec qy; double *qy_p;
#ifdef P4_TO_P8
  Vec qz; double *qz_p;
#endif

  Vec tmp; double *tmp_p;
  ierr = VecDuplicate(phi, &tmp); CHKERRXX(ierr);

  double diag;
  double dxyz[P4EST_DIM];
  get_dxyz_min(p4est, dxyz, NULL, &diag);

  /* init the neighborhood information if needed */
  ngbd->init_neighbors();

  /* compute the normals */
  Vec normal_own[P4EST_DIM];

  if (normal == NULL)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &normal_own[0]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &normal_own[1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est, nodes, &normal_own[2]); CHKERRXX(ierr);
#endif
    compute_normals(*ngbd, phi, normal_own);
  }
  else
  {
    normal_own[0] = normal[0];
    normal_own[1] = normal[1];
#ifdef P4_TO_P8
    normal_own[2] = normal[2];
#endif
  }

  double DIM(*nx, *ny, *nz);
  ierr = VecGetArray(normal_own[0], &nx); CHKERRXX(ierr);
  ierr = VecGetArray(normal_own[1], &ny); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecGetArray(normal_own[2], &nz); CHKERRXX(ierr);
#endif

  ierr = VecGetArray(q , &q_p) ; CHKERRXX(ierr);

  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  /* initialize qn */
  if(order >=1 ) {

    EXECD( ierr = VecCreateGhostNodes(p4est, nodes, &qx); CHKERRXX(ierr),
           ierr = VecCreateGhostNodes(p4est, nodes, &qy); CHKERRXX(ierr),
           ierr = VecCreateGhostNodes(p4est, nodes, &qz); CHKERRXX(ierr) );

    if (q_n == NULL) {
      ierr = VecCreateGhostNodes(p4est, nodes, &qn); CHKERRXX(ierr);
    } else {
      qn = q_n;
    }

    ierr = VecCreateGhostNodes(p4est, nodes, &b_qn_well_defined); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);
    EXECD( ierr = VecGetArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecGetArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecGetArray(qz, &qz_p); CHKERRXX(ierr); );

    for (int map_idx = 0; map_idx < 2; ++map_idx) {
      const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
      for(size_t n_map=0; n_map<map.size(); ++n_map) {
        p4est_locidx_t n = map[n_map];

        if (phi_p[n] > band_use) { // skips if too deep inside the negative domain (specified by band_use)
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];

          if (qnnn.is_stencil_in_negative_domain(mask_p)) {
            // use central differences if all neighbors have well defined values
            b_qn_well_defined_p[n] = -1;
            EXECD( qx_p[n] = qnnn.dx_central(q_p),
                   qy_p[n] = qnnn.dy_central(q_p),
                   qz_p[n] = qnnn.dz_central(q_p) );

            qn_p[n] = SUMD(nx[n]*qx_p[n], ny[n]*qy_p[n], nz[n]*qz_p[n]);
          } else if (mask_p[qnnn.node_000]<-EPS && bc != NULL) {
            // we process separately the case when the center node has a well defined value but not all neighbors
            // however pointwise boundary conditions (Dirichlet) are provided

            // default calculations in case the following correction fails
            b_qn_well_defined_p[n] = 1;

            if (bc->type == DIRICHLET && bc->pointwise && bc->num_value_pts(n) < P4EST_DIM+1 && bc->num_value_pts(n) > 0) {
              // get stencil arms
              double DIMPM( d_m00 = qnnn.d_m00, d_p00 = qnnn.d_p00,
                            d_0m0 = qnnn.d_0m0, d_0p0 = qnnn.d_0p0,
                            d_00m = qnnn.d_00m, d_00p = qnnn.d_00p );

              // check which neighbors have well defined values
              bool DIMPM( nei_m00 = mask_p[qnnn.neighbor_m00()] < -EPS, nei_p00 = mask_p[qnnn.neighbor_p00()] < -EPS,
                          nei_0m0 = mask_p[qnnn.neighbor_0m0()] < -EPS, nei_0p0 = mask_p[qnnn.neighbor_0p0()] < -EPS,
                          nei_00m = mask_p[qnnn.neighbor_00m()] < -EPS, nei_00p = mask_p[qnnn.neighbor_00p()] < -EPS );

              // get values at grid nodes (assuming grid is uniform near the interface)
              double q_000 = q_p[n];
              double DIMPM( q_m00 = qnnn.f_m00_linear(q_p), q_p00 = qnnn.f_p00_linear(q_p),
                            q_0m0 = qnnn.f_0m0_linear(q_p), q_0p0 = qnnn.f_0p0_linear(q_p),
                            q_00m = qnnn.f_00m_linear(q_p), q_00p = qnnn.f_00p_linear(q_p) );

              // adjust stencil arms and values near interface
              double d_min = diag;

              if (phi_p[n] < 0) { // regular Shortley-Weller

                for (unsigned int i = 0; i < bc->num_value_pts(n); ++i) {
                  int idx = bc->idx_value_pt(n,i);
                  interface_point_cartesian_t *pt = &bc->dirichlet_pts[idx];
                  switch (pt->dir)
                  {
                    case dir::f_m00: d_m00 = pt->dist; q_m00 = bc->get_value_pw(n,i); nei_m00 = true; break;
                    case dir::f_p00: d_p00 = pt->dist; q_p00 = bc->get_value_pw(n,i); nei_p00 = true; break;

                    case dir::f_0m0: d_0m0 = pt->dist; q_0m0 = bc->get_value_pw(n,i); nei_0m0 = true; break;
                    case dir::f_0p0: d_0p0 = pt->dist; q_0p0 = bc->get_value_pw(n,i); nei_0p0 = true; break;
#ifdef P4_TO_P8
                    case dir::f_00m: d_00m = pt->dist; q_00m = bc->get_value_pw(n,i); nei_00m = true; break;
                    case dir::f_00p: d_00p = pt->dist; q_00p = bc->get_value_pw(n,i); nei_00p = true; break;
#endif
                  }
                  d_min = MIN(d_min, pt->dist);
                }

              } else { // extended Shortley-Weller

                for (unsigned int i = 0; i < bc->num_value_pts(n); ++i) {
                  int idx = bc->idx_value_pt(n,i);
                  interface_point_cartesian_t *pt = &bc->dirichlet_pts[idx];
                  switch (pt->dir) {
                    case dir::f_p00: d_m00 = -pt->dist; q_m00 = bc->get_value_pw(n,i); nei_m00 = true; break;
                    case dir::f_m00: d_p00 = -pt->dist; q_p00 = bc->get_value_pw(n,i); nei_p00 = true; break;

                    case dir::f_0p0: d_0m0 = -pt->dist; q_0m0 = bc->get_value_pw(n,i); nei_0m0 = true; break;
                    case dir::f_0m0: d_0p0 = -pt->dist; q_0p0 = bc->get_value_pw(n,i); nei_0p0 = true; break;
#ifdef P4_TO_P8
                    case dir::f_00p: d_00m = -pt->dist; q_00m = bc->get_value_pw(n,i); nei_00m = true; break;
                    case dir::f_00m: d_00p = -pt->dist; q_00p = bc->get_value_pw(n,i); nei_00p = true; break;
#endif
                  }
                  d_min = MIN(d_min, pt->dist);
                }
              }

              bool well_defined = ANDD(nei_m00 && nei_p00,
                                       nei_0m0 && nei_0p0,
                                       nei_00m && nei_00p );

              if (d_min > bc_rel_thresh*diag && well_defined)
              {
                b_qn_well_defined_p[n] = -1;

                EXECD( qx_p[n] = ((q_p00-q_000)*d_m00/d_p00 + (q_000-q_m00)*d_p00/d_m00)/(d_m00+d_p00),
                       qy_p[n] = ((q_0p0-q_000)*d_0m0/d_0p0 + (q_000-q_0m0)*d_0p0/d_0m0)/(d_0m0+d_0p0),
                       qz_p[n] = ((q_00p-q_000)*d_00m/d_00p + (q_000-q_00m)*d_00p/d_00m)/(d_00m+d_00p) );
              }
            }

            if (b_qn_well_defined_p[n] < 0) {
              qn_p[n] = SUMD(nx[n]*qx_p[n],
                             ny[n]*qy_p[n],
                             nz[n]*qz_p[n]);
            } else {
              if (use_nonzero_guess) {
                qn_p[n] = SUMD( nx[n]*qnnn.dx_central(q_p),
                                ny[n]*qnnn.dy_central(q_p),
                                nz[n]*qnnn.dz_central(q_p) );
              } else if (q_n == NULL) {
                qn_p[n] = 0;
              }

              EXECD( qx_p[n] = 0,
                     qy_p[n] = 0,
                     qz_p[n] = 0 );
            }

          } else {
            b_qn_well_defined_p[n] = 1;

            if (use_nonzero_guess) {
              qn_p[n] = SUMD( nx[n]*qnnn.dx_central(q_p),
                              ny[n]*qnnn.dy_central(q_p),
                              nz[n]*qnnn.dz_central(q_p) );
            } else if (q_n == NULL) {
              qn_p[n] = 0;
            }

            EXECD( qx_p[n] = 0,
                   qy_p[n] = 0,
                   qz_p[n] = 0 );
          }

        } else {
          b_qn_well_defined_p[n] = -1;
          qn_p[n] = 0;
          EXECD( qx_p[n] = 0,
                 qy_p[n] = 0,
                 qz_p[n] = 0 );
        }
      }

      if (map_idx == 0) { // initiate the communication
        ierr = VecGhostUpdateBegin(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(qn,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        EXECD( ierr = VecGhostUpdateBegin(qx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
               ierr = VecGhostUpdateBegin(qy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
               ierr = VecGhostUpdateBegin(qz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr) );

      } else { // end update communication
        ierr = VecGhostUpdateEnd(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(qn,  INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        EXECD( ierr = VecGhostUpdateEnd(qx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
               ierr = VecGhostUpdateEnd(qy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
               ierr = VecGhostUpdateEnd(qz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr) );
      }
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    EXECD( ierr = VecRestoreArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qz, &qz_p); CHKERRXX(ierr) );
  }

  /* initialize qnn */
  if(order == 2) {
    if (q_nn == NULL) {
      ierr = VecCreateGhostNodes(p4est, nodes, &qnn); CHKERRXX(ierr);
    } else {
      qnn = q_nn;
    }

    ierr = VecDuplicate(b_qn_well_defined, &b_qnn_well_defined); CHKERRXX(ierr);

    EXECD( ierr = VecGetArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecGetArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecGetArray(qz, &qz_p); CHKERRXX(ierr) );

    ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecGetArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    for (int map_idx = 0; map_idx < 2; ++map_idx) {
      const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
      for(size_t n_map=0; n_map<map.size(); ++n_map) {
        p4est_locidx_t n = map[n_map];

        if (phi_p[n] > band_use) {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          b_qnn_well_defined_p[n] = 1;
          if (qnnn.is_stencil_in_negative_domain(mask_p)) {
            // if values for standard stencil are well defined we know we will be able to compute
            // ``pure'' second derivatives (i.e., xx, yy, and zz)
            // however we don't know whether cross derivatives are well defined
            // so we first try to compute cross derivatives
            double Qxy = 0;
#ifdef P4_TO_P8
            double Qyz = 0;
            double Qzx = 0;
#endif
            if (qnnn.is_stencil_in_negative_domain(b_qn_well_defined_p)) {
              // simple case: all necessary first derivatives are well-defined
              b_qnn_well_defined_p[n] = -1;

              Qxy = .5*(qnnn.dx_central(qy_p) + qnnn.dy_central(qx_p));
#ifdef P4_TO_P8
              Qyz = .5*(qnnn.dy_central(qz_p) + qnnn.dz_central(qy_p));
              Qzx = .5*(qnnn.dz_central(qx_p) + qnnn.dx_central(qz_p));
#endif
            } else {
              // if simple case fails, we try to compute cross derivatives directly
              p4est_locidx_t neighbors       [num_neighbors_cube];
              bool           neighbors_exists[num_neighbors_cube];

              ngbd->get_all_neighbors(n, neighbors, neighbors_exists);

              bool all_neighbors_exist =
                  CODE3D( neighbors_exists[nn_m0m] && neighbors_exists[nn_m0p] && neighbors_exists[nn_p0m] && neighbors_exists[nn_p0p] &&
                          neighbors_exists[nn_0mm] && neighbors_exists[nn_0pm] && neighbors_exists[nn_0mp] && neighbors_exists[nn_0pp] && )
                          neighbors_exists[nn_mm0] && neighbors_exists[nn_pm0] && neighbors_exists[nn_mp0] && neighbors_exists[nn_pp0];

              if (all_neighbors_exist)
              {
                bool all_neighbors_in_egative_domain =
                    CODE3D( mask_p[neighbors[nn_m0m]]<0 && mask_p[neighbors[nn_m0p]]<0 && mask_p[neighbors[nn_p0m]]<0 && mask_p[neighbors[nn_p0p]]<0 &&
                            mask_p[neighbors[nn_0mm]]<0 && mask_p[neighbors[nn_0pm]]<0 && mask_p[neighbors[nn_0mp]]<0 && mask_p[neighbors[nn_0pp]]<0 && )
                            mask_p[neighbors[nn_mm0]]<0 && mask_p[neighbors[nn_pm0]]<0 && mask_p[neighbors[nn_mp0]]<0 && mask_p[neighbors[nn_pp0]]<0;

                if (all_neighbors_in_egative_domain) {
                  Qxy = (q_p[neighbors[nn_pp0]] + q_p[neighbors[nn_mm0]] - q_p[neighbors[nn_mp0]] - q_p[neighbors[nn_pm0]])/(qnnn.d_m00+qnnn.d_p00)/(qnnn.d_0m0+qnnn.d_0p0);
#ifdef P4_TO_P8
                  Qyz = (q_p[neighbors[nn_0pp]] + q_p[neighbors[nn_0mm]] - q_p[neighbors[nn_0mp]] - q_p[neighbors[nn_0pm]])/(qnnn.d_0m0+qnnn.d_0p0)/(qnnn.d_00m+qnnn.d_00p);
                  Qzx = (q_p[neighbors[nn_p0p]] + q_p[neighbors[nn_m0m]] - q_p[neighbors[nn_p0m]] - q_p[neighbors[nn_m0p]])/(qnnn.d_00m+qnnn.d_00p)/(qnnn.d_m00+qnnn.d_p00);
#endif
                  b_qnn_well_defined_p[n] = -1;
                }
              }
            }

            // if cross derivatives are successfully computed we go ahead and compute ``pure'' derivatives
            if (b_qnn_well_defined_p[n] < 0)
            {
              double DIM( Qxx = qnnn.dxx_central(q_p),
                          Qyy = qnnn.dyy_central(q_p),
                          Qzz = qnnn.dzz_central(q_p) );

              double DIM( Qx = qx_p[n],
                          Qy = qy_p[n],
                          Qz = qz_p[n] );

              double DIM( Nx = nx[n],
                          Ny = ny[n],
                          Nz = nz[n] );

              double Nxx = qnnn.dx_central(nx);
              double Nyx = qnnn.dx_central(ny);

              double Nxy = qnnn.dy_central(nx);
              double Nyy = qnnn.dy_central(ny);
#ifdef P4_TO_P8
              double Nzx = qnnn.dx_central(nz);
              double Nzy = qnnn.dy_central(nz);
              double Nxz = qnnn.dz_central(nx);
              double Nyz = qnnn.dz_central(ny);
              double Nzz = qnnn.dz_central(nz);

              qnn_p[n] = Nx*(Nx*Qxx + 2.*Ny*Qxy + Qx*Nxx + Qy*Nyx + Qz*Nzx) +
                         Ny*(Ny*Qyy + 2.*Nz*Qyz + Qx*Nxy + Qy*Nyy + Qz*Nzy) +
                         Nz*(Nz*Qzz + 2.*Nx*Qzx + Qx*Nxz + Qy*Nyz + Qz*Nzz);
#else

              qnn_p[n] = Nx*(Nx*Qxx + 2.*Ny*Qxy + Qx*Nxx + Qy*Nyx) +
                         Ny*(Ny*Qyy             + Qx*Nxy + Qy*Nyy);
#endif
            }
          } else {
            if (use_nonzero_guess) {
              qnn_p[n] = SUMD( nx[n]*qnnn.dx_central(qn_p),
                               ny[n]*qnnn.dy_central(qn_p),
                               nz[n]*qnnn.dz_central(qn_p) );
            } else if (q_nn == NULL) {
              qnn_p[n] = 0;
            }
          }
        } else {
          b_qnn_well_defined_p[n] = -1;
          qnn_p[n] = 0;
        }
      }

      if (map_idx == 0) { // initiate the communication
        ierr = VecGhostUpdateBegin(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      } else { // end update communication
        ierr = VecGhostUpdateEnd(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
    }

    EXECD( ierr = VecRestoreArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qz, &qz_p); CHKERRXX(ierr) );

    ierr = VecRestoreArray(qn , &qn_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

    ierr = VecRestoreArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  if (order >= 1) {
    EXECD( ierr = VecDestroy(qx);  CHKERRXX(ierr),
           ierr = VecDestroy(qy);  CHKERRXX(ierr),
           ierr = VecDestroy(qz);  CHKERRXX(ierr) );
  }

  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);

  /* extrapolate qnn */
  if (order == 2) {
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    ierr = VecCopyGhost(qnn, tmp); CHKERRXX(ierr);

    for (int it = 0; it < iterations; ++it) {

      ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      for (int map_idx = 0; map_idx < 2; ++map_idx) {
        const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
        for(size_t n_map=0; n_map<map.size(); ++n_map) {
          p4est_locidx_t n = map[n_map];

          if (b_qnn_well_defined_p[n] > 0 && phi_p[n] < band_extend)
          {
            const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
            double dt = MIN( MIN( DIM( fabs(qnnn.d_m00), fabs(qnnn.d_0m0), fabs(qnnn.d_00m) ) ),
                             MIN( DIM( fabs(qnnn.d_p00), fabs(qnnn.d_0p0), fabs(qnnn.d_00p) ) ) )
                        / double(P4EST_DIM);

            /* first order one sided derivative */
            double qnnx = nx[n]>0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                  : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
            double qnny = ny[n]>0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                  : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
            double qnnz = nz[n]>0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                  : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
            tmp_p[n] = qnn_p[n] - dt*SUMD( nx[n]*qnnx, ny[n]*qnny, nz[n]*qnnz );
          }
        }

        if (map_idx == 0) { // initiate the communication
          ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        } else { // end update communication
          ierr = VecGhostUpdateEnd(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        }
      }

      ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

      // swap pointers instead of swapping data
      Vec swap_tmp;
      swap_tmp = tmp; tmp = qnn; qnn = swap_tmp;
    }

    // transfer data from temporary variable if there was an odd number of pointer swaps
    if (iterations%2 == 1) {
      Vec swap_tmp;
      swap_tmp = tmp; tmp = qnn; qnn = swap_tmp;

      ierr = VecCopyGhost(tmp, qnn); CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  /* extrapolate qn */
  if (order >= 1) {
    if (order == 2) {
      ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
    }

    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    ierr = VecCopyGhost(qn, tmp); CHKERRXX(ierr);

    for (int it = 0; it < iterations; ++it) {

      ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      for (int map_idx = 0; map_idx < 2; ++map_idx) {
        const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
        for(size_t n_map=0; n_map<map.size(); ++n_map) {
          p4est_locidx_t n = map[n_map];

          if (b_qn_well_defined_p[n]>0 && phi_p[n] < band_extend) {
            const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
            double dt = MIN( MIN( DIM( fabs(qnnn.d_m00), fabs(qnnn.d_0m0), fabs(qnnn.d_00m) ) ),
                             MIN( DIM( fabs(qnnn.d_p00), fabs(qnnn.d_0p0), fabs(qnnn.d_00p) ) ) )
                        / double(P4EST_DIM);

            /* first order one sided derivative */
            double qnx = nx[n]>0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                                 : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
            double qny = ny[n]>0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                                 : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
            double qnz = nz[n]>0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                                 : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
            tmp_p[n] = qn_p[n] - dt*(SUMD( nx[n]*qnx, ny[n]*qny, nz[n]*qnz ) - (order==2 ? qnn_p[n] : 0));
          }
        }

        if (map_idx == 0){ // initiate the communication
          ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        } else { // end update communication
          ierr = VecGhostUpdateEnd(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        }
      }

      ierr = VecRestoreArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

      // swap pointers instead of swapping data
      Vec swap_tmp;
      swap_tmp = tmp; tmp = qn; qn = swap_tmp;
    }

    // transfer data from temporary variable if there was an odd number of pointer swaps
    if (iterations%2 == 1) {
      Vec swap_tmp;
      swap_tmp = tmp; tmp = qn; qn = swap_tmp;

      ierr = VecCopyGhost(tmp, qn); CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    if (order == 2)
    {
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);
      if (q_nn == NULL) { ierr = VecDestroy(qnn); CHKERRXX(ierr); }
    }
  }

  if (order >= 1) { ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr); }
  if (order == 2) { ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr); }

  /* extrapolate q */
  Vec DIM(qxx, qyy, qzz);
  double DIM(*qxx_p, *qyy_p, *qzz_p);

  EXECD( ierr = VecCreateGhostNodes(p4est, nodes, &qxx); CHKERRXX(ierr),
         ierr = VecCreateGhostNodes(p4est, nodes, &qyy); CHKERRXX(ierr),
         ierr = VecCreateGhostNodes(p4est, nodes, &qzz); CHKERRXX(ierr) );

  if (order >= 1) { ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr); }

  EXECD( ierr = VecSetGhost(qxx, 0.); CHKERRXX(ierr),
         ierr = VecSetGhost(qyy, 0.); CHKERRXX(ierr),
         ierr = VecSetGhost(qzz, 0.); CHKERRXX(ierr) );

  ierr = VecCopyGhost(q, tmp); CHKERRXX(ierr);

  for (int it = 0; it < iterations; ++it) {

    ngbd->second_derivatives_central(q, DIM(qxx, qyy, qzz));

    EXECD( ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr),
           ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr),
           ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr) );

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    for (int map_idx = 0; map_idx < 2; ++map_idx) {
      const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
      for(size_t n_map=0; n_map<map.size(); ++n_map) {
        p4est_locidx_t n = map[n_map];

        if (mask_p[n] > -EPS && phi_p[n] < band_extend) {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          double dt = MIN( MIN( DIM( fabs(qnnn.d_m00), fabs(qnnn.d_0m0), fabs(qnnn.d_00m) ) ),
                           MIN( DIM( fabs(qnnn.d_p00), fabs(qnnn.d_0p0), fabs(qnnn.d_00p) ) ) )
                      / double(P4EST_DIM);

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
          if (order == 2)
          {
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

            if (nx[n] < 0) qx -= .5*qnnn.d_p00*qxx_p00;
            else           qx += .5*qnnn.d_m00*qxx_m00;

            if (ny[n] < 0) qy -= .5*qnnn.d_0p0*qyy_0p0;
            else           qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
            if (nz[n] < 0) qz -= .5*qnnn.d_00p*qzz_00p;
            else           qz += .5*qnnn.d_00m*qzz_00m;
#endif
          }

          double change_loc = dt*(SUMD( nx[n]*qx, ny[n]*qy, nz[n]*qz ) - (order>=1 ? qn_p[n] : 0));
          tmp_p[n] = q_p[n] - change_loc;
        }
      }

      if (map_idx == 0) { // initiate the communication
        ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      } else { // end update communication
        ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
    }

    EXECD( ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr) );
    ierr = VecRestoreArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

    // option to do an RK2
    if (use_two_step_extrapolation && order == 2)
    {
      Vec swap=q; q=tmp; tmp=swap;

      ngbd->second_derivatives_central(q, DIM(qxx, qyy, qzz));

      EXECD( ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr),
             ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr),
             ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr) );

      ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      for (int map_idx = 0; map_idx < 2; ++map_idx) {
        const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
        for (size_t n_map=0; n_map<map.size(); ++n_map) {
          p4est_locidx_t n = map[n_map];

          if (mask_p[n] > -EPS && phi_p[n] < band_extend) {
            const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
            double dt = MIN( MIN( DIM( fabs(qnnn.d_m00), fabs(qnnn.d_0m0), fabs(qnnn.d_00m) ) ),
                             MIN( DIM( fabs(qnnn.d_p00), fabs(qnnn.d_0p0), fabs(qnnn.d_00p) ) ) )
                        / double(P4EST_DIM);

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
            qxx_m00 = MINMOD(qxx_p[n], qxx_m00);
            qxx_p00 = MINMOD(qxx_p[n], qxx_p00);

            qyy_0m0 = MINMOD(qyy_p[n], qyy_0m0);
            qyy_0p0 = MINMOD(qyy_p[n], qyy_0p0);
#ifdef P4_TO_P8
            qzz_00m = MINMOD(qzz_p[n], qzz_00m);
            qzz_00p = MINMOD(qzz_p[n], qzz_00p);
#endif

            if(nx[n]<0) qx -= .5*qnnn.d_p00*qxx_p00;
            else        qx += .5*qnnn.d_m00*qxx_m00;

            if(ny[n]<0) qy -= .5*qnnn.d_0p0*qyy_0p0;
            else        qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
            if(nz[n]<0) qz -= .5*qnnn.d_00p*qzz_00p;
            else        qz += .5*qnnn.d_00m*qzz_00m;
#endif

            double change_loc = dt*SUMD( nx[n]*qx, ny[n]*qy, nz[n]*qz ) - (order>=1 ? dt*qn_p[n] : 0);

            change_loc = .5*(q_p[n] - change_loc - tmp_p[n]);
            tmp_p[n] = tmp_p[n] + change_loc;
          } else {
            tmp_p[n] = q_p[n];
          }
        }

        if (map_idx == 0) { // initiate the communication
          ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        } else { // end update communication
          ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        }
      }

      EXECD( ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr),
             ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr),
             ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr) );
      ierr = VecRestoreArray(q  , &q_p  ); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

      // swaping pointers instead of physically copying data
      swap=q; q=tmp; tmp=swap;
    } else {
      Vec swap_tmp = tmp; tmp = q; q = swap_tmp;
    }
  }

  // transfer data from temporary variable if there was an odd number of pointer swaps
  if (iterations%2 == 1 && !(use_two_step_extrapolation && order == 2)) {
    Vec swap_tmp = tmp; tmp = q; q = swap_tmp;
    ierr = VecCopyGhost(tmp, q); CHKERRXX(ierr);
  }

  // clean up and restore pointers
  if (order >= 1) {
    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);
    if (q_n == NULL) { ierr = VecDestroy(qn); CHKERRXX(ierr); }
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  if (mask != NULL) {
    ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
  }

  EXECD( ierr = VecDestroy(qxx); CHKERRXX(ierr),
         ierr = VecDestroy(qyy); CHKERRXX(ierr),
         ierr = VecDestroy(qzz); CHKERRXX(ierr) );

  ierr = VecDestroy(tmp); CHKERRXX(ierr);

  ierr = VecRestoreArray(normal_own[0], &nx); CHKERRXX(ierr);
  ierr = VecRestoreArray(normal_own[1], &ny); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(normal_own[2], &nz); CHKERRXX(ierr);
#endif

  if (normal == NULL)
  {
    ierr = VecDestroy(normal_own[0]); CHKERRXX(ierr);
    ierr = VecDestroy(normal_own[1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(normal_own[2]); CHKERRXX(ierr);
#endif
  }

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_over_interface_TVD, phi, q, 0, 0); CHKERRXX(ierr);
}


void my_p4est_level_set_t::extend_Over_Interface_TVD_Full(Vec phi, Vec q, int iterations, int order,
                                                          double band_use, double band_extend,
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

  // band_use is assumed to be negative but user can specify a positive value
  if (band_use > 0) band_use = -band_use;

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

  double diag;
  double dxyz[P4EST_DIM];
  get_dxyz_min(p4est, dxyz, NULL, &diag);

  /* init the neighborhood information if needed */
  ngbd->init_neighbors();

  /* compute the normals */
  double DIM(*nx, *ny, *nz);

  if (normal != NULL) {
    EXECD( ierr = VecGetArray(normal[0], &nx); CHKERRXX(ierr),
           ierr = VecGetArray(normal[1], &ny); CHKERRXX(ierr),
           ierr = VecGetArray(normal[2], &nz); CHKERRXX(ierr) );
  } else {
    EXECD( nx = new double[nodes->num_owned_indeps],
           ny = new double[nodes->num_owned_indeps],
           nz = new double[nodes->num_owned_indeps] );

    foreach_local_node(n, nodes) {
      if (phi_p[n] > band_use && phi_p[n] < band_extend) { // calculate only in the region where it will be used
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];

        EXECD( nx[n] = qnnn.dx_central(phi_p),
               ny[n] = qnnn.dy_central(phi_p),
               nz[n] = qnnn.dz_central(phi_p) );

        double norm = ABSD(nx[n], ny[n], nz[n]);
        if (norm > EPS) {
          EXECD( nx[n] /= norm,
                 ny[n] /= norm,
                 nz[n] /= norm );
        } else {
          EXECD( nx[n] = 0,
                 ny[n] = 0,
                 nz[n] = 0 );
        }
      }
    }
  }
  // note that we do not need to update ghost values of normals

  ierr = VecGetArray(q , &q_p) ; CHKERRXX(ierr);

  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  /* initialize first order derivatives */
  if (order >=1 ) {

    if (q_d == NULL) {
      EXECD( ierr = VecCreateGhostNodes(p4est, nodes, &qx); CHKERRXX(ierr),
             ierr = VecCreateGhostNodes(p4est, nodes, &qy); CHKERRXX(ierr),
             ierr = VecCreateGhostNodes(p4est, nodes, &qz); CHKERRXX(ierr) );
    } else {
      EXECD( qx = q_d[0],
             qy = q_d[1],
             qz = q_d[2] );
    }

    EXECD( ierr = VecGetArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecGetArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecGetArray(qz, &qz_p); CHKERRXX(ierr); );

    ierr = VecCreateGhostNodes(p4est, nodes, &b_qn_well_defined); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for (int map_idx = 0; map_idx < 2; ++map_idx) {
      const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
      for(size_t n_map=0; n_map<map.size(); ++n_map) {
        p4est_locidx_t n = map[n_map];

        if (phi_p[n] > band_use) { // skips if too deep inside the negative domain (specified by band_use)
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];

          if (qnnn.is_stencil_in_negative_domain(mask_p)) {
            // use central differences if all neighbors have well defined values
            b_qn_well_defined_p[n] = -1;
            EXECD( qx_p[n] = qnnn.dx_central(q_p),
                   qy_p[n] = qnnn.dy_central(q_p),
                   qz_p[n] = qnnn.dz_central(q_p) );

          } else if (mask_p[qnnn.node_000]<-EPS && bc != NULL) {
            // we process separately the case when the center node has a well defined value but not all neighbors
            // however pointwise boundary conditions (Dirichlet) are provided

            // default calculations in case the following correction fails
            b_qn_well_defined_p[n] = 1;
            if (use_nonzero_guess) {
              EXECD( qx_p[n] = qnnn.dx_central(q_p),
                     qy_p[n] = qnnn.dy_central(q_p),
                     qz_p[n] = qnnn.dz_central(q_p) );
            } else if (q_d == NULL) {
              EXECD( qx_p[n] = 0,
                     qy_p[n] = 0,
                     qz_p[n] = 0 );
            }

            // correct for boundary conditions if provided
            if (bc->type == DIRICHLET && bc->pointwise && bc->num_value_pts(n) < P4EST_DIM+1 && bc->num_value_pts(n) > 0) {
              // get stencil arms
              double DIMPM( d_m00 = qnnn.d_m00, d_p00 = qnnn.d_p00,
                            d_0m0 = qnnn.d_0m0, d_0p0 = qnnn.d_0p0,
                            d_00m = qnnn.d_00m, d_00p = qnnn.d_00p );

              // check which neighbors have well defined values
              bool DIMPM( nei_m00 = mask_p[qnnn.neighbor_m00()] < -EPS, nei_p00 = mask_p[qnnn.neighbor_p00()] < -EPS,
                          nei_0m0 = mask_p[qnnn.neighbor_0m0()] < -EPS, nei_0p0 = mask_p[qnnn.neighbor_0p0()] < -EPS,
                          nei_00m = mask_p[qnnn.neighbor_00m()] < -EPS, nei_00p = mask_p[qnnn.neighbor_00p()] < -EPS );

              // get values at grid nodes (assuming grid is uniform near the interface)
              double q_000 = q_p[n];
              double DIMPM( q_m00 = qnnn.f_m00_linear(q_p), q_p00 = qnnn.f_p00_linear(q_p),
                            q_0m0 = qnnn.f_0m0_linear(q_p), q_0p0 = qnnn.f_0p0_linear(q_p),
                            q_00m = qnnn.f_00m_linear(q_p), q_00p = qnnn.f_00p_linear(q_p) );

              // adjust stencil arms and values near interface
              double d_min = diag;

              if (phi_p[n] < 0) { // regular Shortley-Weller

                for (unsigned int i = 0; i < bc->num_value_pts(n); ++i) {
                  int idx = bc->idx_value_pt(n,i);
                  interface_point_cartesian_t *pt = &bc->dirichlet_pts[idx];
                  switch (pt->dir)
                  {
                    case dir::f_m00: d_m00 = pt->dist; q_m00 = bc->get_value_pw(n,i); nei_m00 = true; break;
                    case dir::f_p00: d_p00 = pt->dist; q_p00 = bc->get_value_pw(n,i); nei_p00 = true; break;

                    case dir::f_0m0: d_0m0 = pt->dist; q_0m0 = bc->get_value_pw(n,i); nei_0m0 = true; break;
                    case dir::f_0p0: d_0p0 = pt->dist; q_0p0 = bc->get_value_pw(n,i); nei_0p0 = true; break;
#ifdef P4_TO_P8
                    case dir::f_00m: d_00m = pt->dist; q_00m = bc->get_value_pw(n,i); nei_00m = true; break;
                    case dir::f_00p: d_00p = pt->dist; q_00p = bc->get_value_pw(n,i); nei_00p = true; break;
#endif
                  }
                  d_min = MIN(d_min, pt->dist);
                }

              } else { // extended Shortley-Weller

                for (unsigned int i = 0; i < bc->num_value_pts(n); ++i) {
                  int idx = bc->idx_value_pt(n,i);
                  interface_point_cartesian_t *pt = &bc->dirichlet_pts[idx];
                  switch (pt->dir) {
                    case dir::f_p00: d_m00 = -pt->dist; q_m00 = bc->get_value_pw(n,i); nei_m00 = true; break;
                    case dir::f_m00: d_p00 = -pt->dist; q_p00 = bc->get_value_pw(n,i); nei_p00 = true; break;

                    case dir::f_0p0: d_0m0 = -pt->dist; q_0m0 = bc->get_value_pw(n,i); nei_0m0 = true; break;
                    case dir::f_0m0: d_0p0 = -pt->dist; q_0p0 = bc->get_value_pw(n,i); nei_0p0 = true; break;
#ifdef P4_TO_P8
                    case dir::f_00p: d_00m = -pt->dist; q_00m = bc->get_value_pw(n,i); nei_00m = true; break;
                    case dir::f_00m: d_00p = -pt->dist; q_00p = bc->get_value_pw(n,i); nei_00p = true; break;
#endif
                  }
                  d_min = MIN(d_min, pt->dist);
                }
              }

              bool well_defined = ANDD(nei_m00 && nei_p00,
                                       nei_0m0 && nei_0p0,
                                       nei_00m && nei_00p );

              // use boundary points only if boundary not too close (defined by bc_rel_thresh)
              if (d_min > bc_rel_thresh*diag && well_defined) {
                b_qn_well_defined_p[n] = -1;
                EXECD( qx_p[n] = ((q_p00-q_000)*d_m00/d_p00 + (q_000-q_m00)*d_p00/d_m00)/(d_m00+d_p00),
                       qy_p[n] = ((q_0p0-q_000)*d_0m0/d_0p0 + (q_000-q_0m0)*d_0p0/d_0m0)/(d_0m0+d_0p0),
                       qz_p[n] = ((q_00p-q_000)*d_00m/d_00p + (q_000-q_00m)*d_00p/d_00m)/(d_00m+d_00p) );
              }
            }
          } else {
            b_qn_well_defined_p[n] = 1;
            if (use_nonzero_guess) {
              EXECD( qx_p[n] = qnnn.dx_central(q_p),
                     qy_p[n] = qnnn.dy_central(q_p),
                     qz_p[n] = qnnn.dz_central(q_p) );
            } else if (q_d == NULL) {
              EXECD( qx_p[n] = 0,
                     qy_p[n] = 0,
                     qz_p[n] = 0 );
            }
          }
        } else {
          b_qn_well_defined_p[n] = -1;

          EXECD( qx_p[n] = 0,
                 qy_p[n] = 0,
                 qz_p[n] = 0 );
        }
      }

      if (map_idx == 0) { // initiate the communication
        ierr = VecGhostUpdateBegin(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        EXECD( ierr = VecGhostUpdateBegin(qx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
               ierr = VecGhostUpdateBegin(qy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
               ierr = VecGhostUpdateBegin(qz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr) );
      } else { // end update communication
        ierr = VecGhostUpdateEnd(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        EXECD( ierr = VecGhostUpdateEnd(qx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
               ierr = VecGhostUpdateEnd(qy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
               ierr = VecGhostUpdateEnd(qz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr) );
      }
    }

    EXECD( ierr = VecRestoreArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qz, &qz_p); CHKERRXX(ierr) );

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
  }

  /* initialize second derivatives */
  if(order == 2) {
    if (q_dd == NULL) {
      ierr = VecCreateGhostNodes(p4est, nodes, &qxx); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &qyy); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &qxy); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecCreateGhostNodes(p4est, nodes, &qzz); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &qyz); CHKERRXX(ierr);
      ierr = VecCreateGhostNodes(p4est, nodes, &qzx); CHKERRXX(ierr);
#endif
    } else {
      qxx = q_dd[0];
      qyy = q_dd[1];
      qxy = q_dd[P4EST_DIM];
#ifdef P4_TO_P8
      qzz = q_dd[2];
      qyz = q_dd[4];
      qzx = q_dd[5];
#endif
    }

    ierr = VecDuplicate(b_qn_well_defined, &b_qnn_well_defined); CHKERRXX(ierr);

    EXECD( ierr = VecGetArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecGetArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecGetArray(qz, &qz_p); CHKERRXX(ierr) );

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
    ierr = VecGetArray(qxy, &qxy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyz, &qyz_p); CHKERRXX(ierr);
    ierr = VecGetArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    for (int map_idx = 0; map_idx < 2; ++map_idx) {
      const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
      for(size_t n_map=0; n_map<map.size(); ++n_map) {
        p4est_locidx_t n = map[n_map];

        if (phi_p[n] > band_use) {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          b_qnn_well_defined_p[n] = 1;
          if (qnnn.is_stencil_in_negative_domain(mask_p)) {
            // if values for standard stencil are well defined we know we will be able to compute
            // ``pure'' second derivatives (i.e., xx, yy, and zz)
            // however we don't know whether cross derivatives are well defined
            // so we first try to compute cross derivatives
            if (qnnn.is_stencil_in_negative_domain(b_qn_well_defined_p)) {
              // simple case: all necessary first derivatives are well-defined
              b_qnn_well_defined_p[n] = -1;

              qxy_p[n] = .5*(qnnn.dx_central(qy_p) + qnnn.dy_central(qx_p));
#ifdef P4_TO_P8
              qyz_p[n] = .5*(qnnn.dy_central(qz_p) + qnnn.dz_central(qy_p));
              qzx_p[n] = .5*(qnnn.dz_central(qx_p) + qnnn.dx_central(qz_p));
#endif
            } else {
              // if simple case fails, we try to compute cross derivatives directly
              p4est_locidx_t neighbors       [num_neighbors_cube];
              bool           neighbors_exists[num_neighbors_cube];

              ngbd->get_all_neighbors(n, neighbors, neighbors_exists);

              bool all_neighbors_exist =
                  CODE3D( neighbors_exists[nn_m0m] && neighbors_exists[nn_m0p] && neighbors_exists[nn_p0m] && neighbors_exists[nn_p0p] &&
                          neighbors_exists[nn_0mm] && neighbors_exists[nn_0pm] && neighbors_exists[nn_0mp] && neighbors_exists[nn_0pp] && )
                          neighbors_exists[nn_mm0] && neighbors_exists[nn_pm0] && neighbors_exists[nn_mp0] && neighbors_exists[nn_pp0];

              if (all_neighbors_exist) {
                bool all_neighbors_in_egative_domain =
                    CODE3D( mask_p[neighbors[nn_m0m]]<0 && mask_p[neighbors[nn_m0p]]<0 && mask_p[neighbors[nn_p0m]]<0 && mask_p[neighbors[nn_p0p]]<0 &&
                            mask_p[neighbors[nn_0mm]]<0 && mask_p[neighbors[nn_0pm]]<0 && mask_p[neighbors[nn_0mp]]<0 && mask_p[neighbors[nn_0pp]]<0 && )
                            mask_p[neighbors[nn_mm0]]<0 && mask_p[neighbors[nn_pm0]]<0 && mask_p[neighbors[nn_mp0]]<0 && mask_p[neighbors[nn_pp0]]<0;

                if (all_neighbors_in_egative_domain) {
                  qxy_p[n] = (q_p[neighbors[nn_pp0]] + q_p[neighbors[nn_mm0]] - q_p[neighbors[nn_mp0]] - q_p[neighbors[nn_pm0]])/(qnnn.d_m00+qnnn.d_p00)/(qnnn.d_0m0+qnnn.d_0p0);
#ifdef P4_TO_P8
                  qyz_p[n] = (q_p[neighbors[nn_0pp]] + q_p[neighbors[nn_0mm]] - q_p[neighbors[nn_0mp]] - q_p[neighbors[nn_0pm]])/(qnnn.d_0m0+qnnn.d_0p0)/(qnnn.d_00m+qnnn.d_00p);
                  qzx_p[n] = (q_p[neighbors[nn_p0p]] + q_p[neighbors[nn_m0m]] - q_p[neighbors[nn_p0m]] - q_p[neighbors[nn_m0p]])/(qnnn.d_00m+qnnn.d_00p)/(qnnn.d_m00+qnnn.d_p00);
#endif
                  b_qnn_well_defined_p[n] = -1;
                }
              }
            }

            // if cross derivatives are successfully computed we go ahead and compute ``pure'' derivatives too
            if (b_qnn_well_defined_p[n] < 0) {
              EXECD( qxx_p[n] = qnnn.dxx_central(q_p),
                     qyy_p[n] = qnnn.dyy_central(q_p),
                     qzz_p[n] = qnnn.dzz_central(q_p) );
            }
          } else {
            if (use_nonzero_guess) {
              qxx_p[n] = qnnn.dxx_central(q_p);
              qxy_p[n] = .5*(qnnn.dx_central(qy_p) + qnnn.dy_central(qx_p));
              qyy_p[n] = qnnn.dyy_central(q_p);
#ifdef P4_TO_P8
              qyz_p[n] = .5*(qnnn.dy_central(qz_p) + qnnn.dz_central(qy_p));
              qzz_p[n] = qnnn.dzz_central(q_p);
              qzx_p[n] = .5*(qnnn.dz_central(qx_p) + qnnn.dx_central(qz_p));
#endif
            } else if (q_dd == NULL) {
              qxx_p[n] = 0;
              qxy_p[n] = 0;
              qyy_p[n] = 0;
#ifdef P4_TO_P8
              qyz_p[n] = 0;
              qzz_p[n] = 0;
              qzx_p[n] = 0;
#endif
            }
          }
        } else {
          b_qnn_well_defined_p[n] = -1;

          qxx_p[n] = 0;
          qyy_p[n] = 0;
          qxy_p[n] = 0;
#ifdef P4_TO_P8
          qzz_p[n] = 0;
          qyz_p[n] = 0;
          qzx_p[n] = 0;
#endif
        }
      }

      if (map_idx == 0) { // initiate the communication
        ierr = VecGhostUpdateBegin(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

        ierr = VecGhostUpdateBegin(qxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(qxy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(qyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
        ierr = VecGhostUpdateBegin(qyz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(qzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateBegin(qzx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
      } else { // end update communication
        ierr = VecGhostUpdateEnd(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

        ierr = VecGhostUpdateEnd(qxx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(qxy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(qyy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#ifdef P4_TO_P8
        ierr = VecGhostUpdateEnd(qyz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(qzz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        ierr = VecGhostUpdateEnd(qzx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
#endif
      }
    }

    EXECD( ierr = VecRestoreArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qz, &qz_p); CHKERRXX(ierr) );

    ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qxy, &qxy_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(qyz, &qyz_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif

    ierr = VecRestoreArray(b_qn_well_defined , &b_qn_well_defined_p ); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);

  /* extrapolate second order derivatives */
  if (order == 2) {
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    Vec tmp_xx; double *tmp_xx_p; ierr = VecDuplicate(qxx, &tmp_xx); CHKERRXX(ierr);
    Vec tmp_xy; double *tmp_xy_p; ierr = VecDuplicate(qxy, &tmp_xy); CHKERRXX(ierr);
    Vec tmp_yy; double *tmp_yy_p; ierr = VecDuplicate(qyy, &tmp_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    Vec tmp_yz; double *tmp_yz_p; ierr = VecDuplicate(qyz, &tmp_yz); CHKERRXX(ierr);
    Vec tmp_zz; double *tmp_zz_p; ierr = VecDuplicate(qzz, &tmp_zz); CHKERRXX(ierr);
    Vec tmp_zx; double *tmp_zx_p; ierr = VecDuplicate(qzx, &tmp_zx); CHKERRXX(ierr);
#endif

    VecCopyGhost(qxx, tmp_xx);
    VecCopyGhost(qxy, tmp_xy);
    VecCopyGhost(qyy, tmp_yy);
#ifdef P4_TO_P8
    VecCopyGhost(qyz, tmp_yz);
    VecCopyGhost(qzz, tmp_zz);
    VecCopyGhost(qzx, tmp_zx);
#endif

    for (int it = 0; it < iterations; ++it) {

      ierr = VecGetArray(tmp_xx, &tmp_xx_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_xy, &tmp_xy_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_yy, &tmp_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(tmp_yz, &tmp_yz_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_zz, &tmp_zz_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp_zx, &tmp_zx_p); CHKERRXX(ierr);
#endif

      ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecGetArray(qxy, &qxy_p); CHKERRXX(ierr);
      ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(qyz, &qyz_p); CHKERRXX(ierr);
      ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
      ierr = VecGetArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif

      for (int map_idx = 0; map_idx < 2; ++map_idx) {
        const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
        for (size_t n_map=0; n_map<map.size(); ++n_map) {
          p4est_locidx_t n = map[n_map];

          if (b_qnn_well_defined_p[n] > 0 && phi_p[n] < band_extend) {

            const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
            double dt = MIN( MIN( DIM( fabs(qnnn.d_m00), fabs(qnnn.d_0m0), fabs(qnnn.d_00m) ) ),
                             MIN( DIM( fabs(qnnn.d_p00), fabs(qnnn.d_0p0), fabs(qnnn.d_00p) ) ) )
                        / double(P4EST_DIM);

            /* first order one sided derivative */
            double qxxd_dot_n = 0;
            double qxyd_dot_n = 0;
            double qyyd_dot_n = 0;
#ifdef P4_TO_P8
            double qyzd_dot_n = 0;
            double qzzd_dot_n = 0;
            double qzxd_dot_n = 0;
#endif
            if (nx[n] > 0) {
              qxxd_dot_n += nx[n]*(qxx_p[n] - qnnn.f_m00_linear(qxx_p)) / qnnn.d_m00;
              qxyd_dot_n += nx[n]*(qxy_p[n] - qnnn.f_m00_linear(qxy_p)) / qnnn.d_m00;
              qyyd_dot_n += nx[n]*(qyy_p[n] - qnnn.f_m00_linear(qyy_p)) / qnnn.d_m00;
#ifdef P4_TO_P8
              qyzd_dot_n += nx[n]*(qyz_p[n] - qnnn.f_m00_linear(qyz_p)) / qnnn.d_m00;
              qzzd_dot_n += nx[n]*(qzz_p[n] - qnnn.f_m00_linear(qzz_p)) / qnnn.d_m00;
              qzxd_dot_n += nx[n]*(qzx_p[n] - qnnn.f_m00_linear(qzx_p)) / qnnn.d_m00;
#endif
            } else {
              qxxd_dot_n += nx[n]*(qnnn.f_p00_linear(qxx_p) - qxx_p[n]) / qnnn.d_p00;
              qxyd_dot_n += nx[n]*(qnnn.f_p00_linear(qxy_p) - qxy_p[n]) / qnnn.d_p00;
              qyyd_dot_n += nx[n]*(qnnn.f_p00_linear(qyy_p) - qyy_p[n]) / qnnn.d_p00;
#ifdef P4_TO_P8
              qyzd_dot_n += nx[n]*(qnnn.f_p00_linear(qyz_p) - qyz_p[n]) / qnnn.d_p00;
              qzzd_dot_n += nx[n]*(qnnn.f_p00_linear(qzz_p) - qzz_p[n]) / qnnn.d_p00;
              qzxd_dot_n += nx[n]*(qnnn.f_p00_linear(qzx_p) - qzx_p[n]) / qnnn.d_p00;
#endif
            }

            if (ny[n] > 0) {
              qxxd_dot_n += ny[n]*(qxx_p[n] - qnnn.f_0m0_linear(qxx_p)) / qnnn.d_0m0;
              qxyd_dot_n += ny[n]*(qxy_p[n] - qnnn.f_0m0_linear(qxy_p)) / qnnn.d_0m0;
              qyyd_dot_n += ny[n]*(qyy_p[n] - qnnn.f_0m0_linear(qyy_p)) / qnnn.d_0m0;
#ifdef P4_TO_P8
              qyzd_dot_n += ny[n]*(qyz_p[n] - qnnn.f_0m0_linear(qyz_p)) / qnnn.d_0m0;
              qzzd_dot_n += ny[n]*(qzz_p[n] - qnnn.f_0m0_linear(qzz_p)) / qnnn.d_0m0;
              qzxd_dot_n += ny[n]*(qzx_p[n] - qnnn.f_0m0_linear(qzx_p)) / qnnn.d_0m0;
#endif
            } else {
              qxxd_dot_n += ny[n]*(qnnn.f_0p0_linear(qxx_p) - qxx_p[n]) / qnnn.d_0p0;
              qxyd_dot_n += ny[n]*(qnnn.f_0p0_linear(qxy_p) - qxy_p[n]) / qnnn.d_0p0;
              qyyd_dot_n += ny[n]*(qnnn.f_0p0_linear(qyy_p) - qyy_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
              qyzd_dot_n += ny[n]*(qnnn.f_0p0_linear(qyz_p) - qyz_p[n]) / qnnn.d_0p0;
              qzzd_dot_n += ny[n]*(qnnn.f_0p0_linear(qzz_p) - qzz_p[n]) / qnnn.d_0p0;
              qzxd_dot_n += ny[n]*(qnnn.f_0p0_linear(qzx_p) - qzx_p[n]) / qnnn.d_0p0;
#endif
            }

#ifdef P4_TO_P8
            if (nz[n] > 0) {
              qxxd_dot_n += nz[n]*(qxx_p[n] - qnnn.f_00m_linear(qxx_p)) / qnnn.d_00m;
              qxyd_dot_n += nz[n]*(qxy_p[n] - qnnn.f_00m_linear(qxy_p)) / qnnn.d_00m;
              qyyd_dot_n += nz[n]*(qyy_p[n] - qnnn.f_00m_linear(qyy_p)) / qnnn.d_00m;
              qyzd_dot_n += nz[n]*(qyz_p[n] - qnnn.f_00m_linear(qyz_p)) / qnnn.d_00m;
              qzzd_dot_n += nz[n]*(qzz_p[n] - qnnn.f_00m_linear(qzz_p)) / qnnn.d_00m;
              qzxd_dot_n += nz[n]*(qzx_p[n] - qnnn.f_00m_linear(qzx_p)) / qnnn.d_00m;
            } else {
              qxxd_dot_n += nz[n]*(qnnn.f_00p_linear(qxx_p) - qxx_p[n]) / qnnn.d_00p;
              qxyd_dot_n += nz[n]*(qnnn.f_00p_linear(qxy_p) - qxy_p[n]) / qnnn.d_00p;
              qyyd_dot_n += nz[n]*(qnnn.f_00p_linear(qyy_p) - qyy_p[n]) / qnnn.d_00p;
              qyzd_dot_n += nz[n]*(qnnn.f_00p_linear(qyz_p) - qyz_p[n]) / qnnn.d_00p;
              qzzd_dot_n += nz[n]*(qnnn.f_00p_linear(qzz_p) - qzz_p[n]) / qnnn.d_00p;
              qzxd_dot_n += nz[n]*(qnnn.f_00p_linear(qzx_p) - qzx_p[n]) / qnnn.d_00p;
            }
#endif
            tmp_xx_p[n] = qxx_p[n] - qxxd_dot_n*dt;
            tmp_xy_p[n] = qxy_p[n] - qxyd_dot_n*dt;
            tmp_yy_p[n] = qyy_p[n] - qyyd_dot_n*dt;
#ifdef P4_TO_P8
            tmp_yz_p[n] = qyz_p[n] - qyzd_dot_n*dt;
            tmp_zz_p[n] = qzz_p[n] - qzzd_dot_n*dt;
            tmp_zx_p[n] = qzx_p[n] - qzxd_dot_n*dt;
#endif
          }
        }

        if (map_idx == 0) { // initiate the communication
          ierr = VecGhostUpdateBegin(tmp_xx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(tmp_xy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(tmp_yy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #ifdef P4_TO_P8
          ierr = VecGhostUpdateBegin(tmp_yz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(tmp_zz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateBegin(tmp_zx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #endif
        } else { // end update communication
          ierr = VecGhostUpdateEnd(tmp_xx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(tmp_xy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(tmp_yy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #ifdef P4_TO_P8
          ierr = VecGhostUpdateEnd(tmp_yz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(tmp_zz, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
          ierr = VecGhostUpdateEnd(tmp_zx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    #endif
        }
      }

      ierr = VecRestoreArray(tmp_xx, &tmp_xx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_xy, &tmp_xy_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_yy, &tmp_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(tmp_yz, &tmp_yz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_zz, &tmp_zz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(tmp_zx, &tmp_zx_p); CHKERRXX(ierr);
#endif

      ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qxy, &qxy_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(qyz, &qyz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif

      // swap pointers instead of swapping data
      Vec swap_tmp;
      swap_tmp = tmp_xx; tmp_xx = qxx; qxx = swap_tmp;
      swap_tmp = tmp_xy; tmp_xy = qxy; qxy = swap_tmp;
      swap_tmp = tmp_yy; tmp_yy = qyy; qyy = swap_tmp;
#ifdef P4_TO_P8
      swap_tmp = tmp_yz; tmp_yz = qyz; qyz = swap_tmp;
      swap_tmp = tmp_zz; tmp_zz = qzz; qzz = swap_tmp;
      swap_tmp = tmp_zx; tmp_zx = qzx; qzx = swap_tmp;
#endif
    }

    // transfer data from temporary variable if there was an odd number of pointer swaps
    if (iterations%2 == 1) {
      Vec swap_tmp;
      swap_tmp = tmp_xx; tmp_xx = qxx; qxx = swap_tmp;
      swap_tmp = tmp_xy; tmp_xy = qxy; qxy = swap_tmp;
      swap_tmp = tmp_yy; tmp_yy = qyy; qyy = swap_tmp;
#ifdef P4_TO_P8
      swap_tmp = tmp_yz; tmp_yz = qyz; qyz = swap_tmp;
      swap_tmp = tmp_zz; tmp_zz = qzz; qzz = swap_tmp;
      swap_tmp = tmp_zx; tmp_zx = qzx; qzx = swap_tmp;
#endif

      VecCopyGhost(tmp_xx, qxx);
      VecCopyGhost(tmp_xy, qxy);
      VecCopyGhost(tmp_yy, qyy);
#ifdef P4_TO_P8
      VecCopyGhost(tmp_yz, qyz);
      VecCopyGhost(tmp_zz, qzz);
      VecCopyGhost(tmp_zx, qzx);
#endif
    }

    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    ierr = VecDestroy(tmp_xx); CHKERRXX(ierr);
    ierr = VecDestroy(tmp_xy); CHKERRXX(ierr);
    ierr = VecDestroy(tmp_yy); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecDestroy(tmp_yz); CHKERRXX(ierr);
    ierr = VecDestroy(tmp_zz); CHKERRXX(ierr);
    ierr = VecDestroy(tmp_zx); CHKERRXX(ierr);
#endif
  }

  /* extrapolate qn */
  if (order >= 1) {
    if (order == 2) {
      ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecGetArray(qxy, &qxy_p); CHKERRXX(ierr);
      ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecGetArray(qyz, &qyz_p); CHKERRXX(ierr);
      ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
      ierr = VecGetArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif
    }

    EXECD( Vec tmp_x; double *tmp_x_p; ierr = VecDuplicate(qx, &tmp_x); CHKERRXX(ierr) ,
           Vec tmp_y; double *tmp_y_p; ierr = VecDuplicate(qy, &tmp_y); CHKERRXX(ierr) ,
           Vec tmp_z; double *tmp_z_p; ierr = VecDuplicate(qz, &tmp_z); CHKERRXX(ierr) );

    EXECD( VecCopyGhost(qx, tmp_x) ,
           VecCopyGhost(qy, tmp_y) ,
           VecCopyGhost(qz, tmp_z) );

    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    // precompute right hand side
    std::vector<double> DIM( qxn(nodes->num_owned_indeps, 0),
                             qyn(nodes->num_owned_indeps, 0),
                             qzn(nodes->num_owned_indeps, 0) );

    if (order == 2) {
      foreach_local_node(n, nodes) {
        if (b_qn_well_defined_p[n] > 0 && phi_p[n] < band_extend) {
          EXECD( qxn[n] = SUMD(nx[n]*qxx_p[n], ny[n]*qxy_p[n], nz[n]*qzx_p[n]) ,
                 qyn[n] = SUMD(nx[n]*qxy_p[n], ny[n]*qyy_p[n], nz[n]*qyz_p[n]) ,
                 qzn[n] = SUMD(nx[n]*qzx_p[n], ny[n]*qyz_p[n], nz[n]*qzz_p[n]) );
        }
      }
    }

    for (int it = 0; it < iterations; ++it) {

      EXECD( ierr = VecGetArray(qx, &qx_p); CHKERRXX(ierr) ,
             ierr = VecGetArray(qy, &qy_p); CHKERRXX(ierr) ,
             ierr = VecGetArray(qz, &qz_p); CHKERRXX(ierr) );

      EXECD( ierr = VecGetArray(tmp_x, &tmp_x_p); CHKERRXX(ierr) ,
             ierr = VecGetArray(tmp_y, &tmp_y_p); CHKERRXX(ierr) ,
             ierr = VecGetArray(tmp_z, &tmp_z_p); CHKERRXX(ierr) );

      for (int map_idx = 0; map_idx < 2; ++map_idx) {
        const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
        for(size_t n_map=0; n_map<map.size(); ++n_map) {
          p4est_locidx_t n = map[n_map];

          if (b_qn_well_defined_p[n] > 0 && phi_p[n] < band_extend) {
            const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
            double dt = MIN( MIN( DIM( fabs(qnnn.d_m00), fabs(qnnn.d_0m0), fabs(qnnn.d_00m) ) ),
                             MIN( DIM( fabs(qnnn.d_p00), fabs(qnnn.d_0p0), fabs(qnnn.d_00p) ) ) )
                        / double(P4EST_DIM);

            /* first order one sided derivative */
            double DIM( qxd_dot_n = 0,
                        qyd_dot_n = 0,
                        qzd_dot_n = 0 );

            if (nx[n] > 0) {
              EXECD( qxd_dot_n += nx[n]*(qx_p[n] - qnnn.f_m00_linear(qx_p)) / qnnn.d_m00,
                     qyd_dot_n += nx[n]*(qy_p[n] - qnnn.f_m00_linear(qy_p)) / qnnn.d_m00,
                     qzd_dot_n += nx[n]*(qz_p[n] - qnnn.f_m00_linear(qz_p)) / qnnn.d_m00 );
            } else {
              EXECD( qxd_dot_n += nx[n]*(qnnn.f_p00_linear(qx_p) - qx_p[n]) / qnnn.d_p00,
                     qyd_dot_n += nx[n]*(qnnn.f_p00_linear(qy_p) - qy_p[n]) / qnnn.d_p00,
                     qzd_dot_n += nx[n]*(qnnn.f_p00_linear(qz_p) - qz_p[n]) / qnnn.d_p00 );
            }

            if (ny[n] > 0) {
              EXECD( qxd_dot_n += ny[n]*(qx_p[n] - qnnn.f_0m0_linear(qx_p)) / qnnn.d_0m0,
                     qyd_dot_n += ny[n]*(qy_p[n] - qnnn.f_0m0_linear(qy_p)) / qnnn.d_0m0,
                     qzd_dot_n += ny[n]*(qz_p[n] - qnnn.f_0m0_linear(qz_p)) / qnnn.d_0m0 );
            } else {
              EXECD( qxd_dot_n += ny[n]*(qnnn.f_0p0_linear(qx_p) - qx_p[n]) / qnnn.d_0p0,
                     qyd_dot_n += ny[n]*(qnnn.f_0p0_linear(qy_p) - qy_p[n]) / qnnn.d_0p0,
                     qzd_dot_n += ny[n]*(qnnn.f_0p0_linear(qz_p) - qz_p[n]) / qnnn.d_0p0 );
            }

#ifdef P4_TO_P8
            if (nz[n] > 0) {
              qxd_dot_n += nz[n]*(qx_p[n] - qnnn.f_00m_linear(qx_p)) / qnnn.d_00m;
              qyd_dot_n += nz[n]*(qy_p[n] - qnnn.f_00m_linear(qy_p)) / qnnn.d_00m;
              qzd_dot_n += nz[n]*(qz_p[n] - qnnn.f_00m_linear(qz_p)) / qnnn.d_00m;
            } else {
              qxd_dot_n += nz[n]*(qnnn.f_00p_linear(qx_p) - qx_p[n]) / qnnn.d_00p;
              qyd_dot_n += nz[n]*(qnnn.f_00p_linear(qy_p) - qy_p[n]) / qnnn.d_00p;
              qzd_dot_n += nz[n]*(qnnn.f_00p_linear(qz_p) - qz_p[n]) / qnnn.d_00p;
            }
#endif

            EXECD( tmp_x_p[n] = qx_p[n] - (qxd_dot_n-qxn[n])*dt ,
                   tmp_y_p[n] = qy_p[n] - (qyd_dot_n-qyn[n])*dt ,
                   tmp_z_p[n] = qz_p[n] - (qzd_dot_n-qzn[n])*dt );
          }
        }

        if (map_idx == 0){ // initiate the communication
          EXECD( ierr = VecGhostUpdateBegin(tmp_x, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
                 ierr = VecGhostUpdateBegin(tmp_y, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
                 ierr = VecGhostUpdateBegin(tmp_z, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr) );
        } else { // end update communication
          EXECD( ierr = VecGhostUpdateEnd(tmp_x, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
                 ierr = VecGhostUpdateEnd(tmp_y, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr),
                 ierr = VecGhostUpdateEnd(tmp_z, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr) );
        }
      }

      EXECD( ierr = VecRestoreArray(qx, &qx_p); CHKERRXX(ierr),
             ierr = VecRestoreArray(qy, &qy_p); CHKERRXX(ierr),
             ierr = VecRestoreArray(qz, &qz_p); CHKERRXX(ierr) );

      EXECD( ierr = VecRestoreArray(tmp_x, &tmp_x_p); CHKERRXX(ierr),
             ierr = VecRestoreArray(tmp_y, &tmp_y_p); CHKERRXX(ierr),
             ierr = VecRestoreArray(tmp_z, &tmp_z_p); CHKERRXX(ierr) );

      // swap pointers instead of swapping data
      Vec swap_tmp;
      EXECD( swap_tmp = tmp_x; tmp_x = qx; qx = swap_tmp,
             swap_tmp = tmp_y; tmp_y = qy; qy = swap_tmp,
             swap_tmp = tmp_z; tmp_z = qz; qz = swap_tmp );
    }

    // transfer data from temporary variable if there was an odd number of pointer swaps
    if (iterations%2 == 1) {
      Vec swap_tmp;
      EXECD( swap_tmp = tmp_x; tmp_x = qx; qx = swap_tmp,
             swap_tmp = tmp_y; tmp_y = qy; qy = swap_tmp,
             swap_tmp = tmp_z; tmp_z = qz; qz = swap_tmp );

      EXECD( VecCopyGhost(tmp_x, qx),
             VecCopyGhost(tmp_y, qy),
             VecCopyGhost(tmp_z, qz) );
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    if (order == 2) {
      ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qxy, &qxy_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecRestoreArray(qyz, &qyz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr);
      ierr = VecRestoreArray(qzx, &qzx_p); CHKERRXX(ierr);
#endif
    }

    EXECD( ierr = VecDestroy(tmp_x); CHKERRXX(ierr),
           ierr = VecDestroy(tmp_y); CHKERRXX(ierr),
           ierr = VecDestroy(tmp_z); CHKERRXX(ierr) );
  }

  if (order >= 1) { ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr); }
  if (order == 2) { ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr); }

  /* extrapolate q */
  if (order >= 1) {
    EXECD( ierr = VecGetArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecGetArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecGetArray(qz, &qz_p); CHKERRXX(ierr) );
  }

  if (order == 2) {
    EXECD( ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr),
           ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr),
           ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr) );
  }

  Vec tmp; double *tmp_p; ierr = VecDuplicate(q, &tmp); CHKERRXX(ierr);
  VecCopyGhost(q, tmp);

  for (int it = 0; it < iterations; ++it) {

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    // precompute right hand-side
    std::vector<double> qn(nodes->num_owned_indeps, 0);

    if (order >= 1) {
      foreach_local_node(n, nodes) {
        if (mask_p[n] > -EPS && phi_p[n] < band_extend) {
          qn[n] = SUMD(nx[n]*qx_p[n], ny[n]*qy_p[n], nz[n]*qz_p[n]);
        }
      }
    }

    if (order == 2) {
      foreach_local_node(n, nodes) {
        if (mask_p[n] > -EPS && phi_p[n] < band_extend) {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];

          double qxx_m00 = qnnn.f_m00_linear(qxx_p);
          double qxx_p00 = qnnn.f_p00_linear(qxx_p);

          double qyy_0m0 = qnnn.f_0m0_linear(qyy_p);
          double qyy_0p0 = qnnn.f_0p0_linear(qyy_p);
#ifdef P4_TO_P8
          double qzz_00m = qnnn.f_00m_linear(qzz_p);
          double qzz_00p = qnnn.f_00p_linear(qzz_p);
#endif

          /* minmod operation */
          qxx_m00 = MINMOD(qxx_p[n], qxx_m00);
          qxx_p00 = MINMOD(qxx_p[n], qxx_p00);

          qyy_0m0 = MINMOD(qyy_p[n], qyy_0m0);
          qyy_0p0 = MINMOD(qyy_p[n], qyy_0p0);
#ifdef P4_TO_P8
          qzz_00m = MINMOD(qzz_p[n], qzz_00m);
          qzz_00p = MINMOD(qzz_p[n], qzz_00p);
#endif

          if (nx[n]<0) qn[n] += nx[n]*.5*qnnn.d_p00*qxx_p00;
          else         qn[n] -= nx[n]*.5*qnnn.d_m00*qxx_m00;

          if (ny[n]<0) qn[n] += ny[n]*.5*qnnn.d_0p0*qyy_0p0;
          else         qn[n] -= ny[n]*.5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
          if (nz[n]<0) qn[n] += nz[n]*.5*qnnn.d_00p*qzz_00p;
          else         qn[n] -= nz[n]*.5*qnnn.d_00m*qzz_00m;
#endif
        }
      }
    }

    for (int map_idx = 0; map_idx < 2; ++map_idx) {
      const std::vector<p4est_locidx_t>& map = map_idx == 0 ? layer_nodes : local_nodes;
      for(size_t n_map=0; n_map<map.size(); ++n_map) {
        p4est_locidx_t n = map[n_map];

        if (mask_p[n] > -EPS && phi_p[n] < band_extend) {
          const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
          double dt = MIN( MIN( DIM( fabs(qnnn.d_m00), fabs(qnnn.d_0m0), fabs(qnnn.d_00m) ) ),
                           MIN( DIM( fabs(qnnn.d_p00), fabs(qnnn.d_0p0), fabs(qnnn.d_00p) ) ) )
                      / double(P4EST_DIM);

          /* first order one sided derivatives */
          double Qx = nx[n]>0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                              :-(q_p[n] - qnnn.f_p00_linear(q_p)) / qnnn.d_p00;

          double Qy = ny[n]>0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                              :-(q_p[n] - qnnn.f_0p0_linear(q_p)) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double Qz = nz[n]>0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
                              :-(q_p[n] - qnnn.f_00p_linear(q_p)) / qnnn.d_00p;
#endif

          tmp_p[n] = q_p[n] - (SUMD(nx[n]*Qx, ny[n]*Qy, nz[n]*Qz) - qn[n])*dt;
        }
      }

      if (map_idx == 0) { // initiate the communication
        ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      } else { // end update communication
        ierr = VecGhostUpdateEnd  (tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      }
    }

    ierr = VecRestoreArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecRestoreArray(tmp, &tmp_p); CHKERRXX(ierr);

    // swaping pointers instead of physically copying data
    Vec swap_tmp = tmp; tmp = q; q = swap_tmp;
  }

  // transfer data from temporary variable if there was an odd number of pointer swaps
  if (iterations%2 == 1) {
    Vec swap_tmp = tmp; tmp = q; q = swap_tmp;
    VecCopyGhost(tmp, q);
  }

  ierr = VecDestroy(tmp); CHKERRXX(ierr);

  // clean up and restore pointers
  if(order>=1) {
    EXECD( ierr = VecRestoreArray(qx, &qx_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qy, &qy_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qz, &qz_p); CHKERRXX(ierr) );

    if (q_d == NULL) {
      EXECD( ierr = VecDestroy(qx); CHKERRXX(ierr),
             ierr = VecDestroy(qy); CHKERRXX(ierr),
             ierr = VecDestroy(qz); CHKERRXX(ierr) );
    }
  }

  if (order == 2 && q_dd == NULL) {
    EXECD( ierr = VecRestoreArray(qxx, &qxx_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qyy, &qyy_p); CHKERRXX(ierr),
           ierr = VecRestoreArray(qzz, &qzz_p); CHKERRXX(ierr) );

    if (q_dd == NULL) {
      ierr = VecDestroy(qxx); CHKERRXX(ierr);
      ierr = VecDestroy(qxy); CHKERRXX(ierr);
      ierr = VecDestroy(qyy); CHKERRXX(ierr);
#ifdef P4_TO_P8
      ierr = VecDestroy(qyz); CHKERRXX(ierr);
      ierr = VecDestroy(qzz); CHKERRXX(ierr);
      ierr = VecDestroy(qzx); CHKERRXX(ierr);
#endif
    }
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  if (mask != NULL) {
    ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
  }

  if (normal != NULL) {
    EXECD( ierr = VecRestoreArray(normal[0], &nx); CHKERRXX(ierr),
           ierr = VecRestoreArray(normal[1], &ny); CHKERRXX(ierr),
           ierr = VecRestoreArray(normal[2], &nz); CHKERRXX(ierr) );
  } else {
    EXECD( delete[] nx,
           delete[] ny,
           delete[] nz );
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
  for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);
    qnnn.gradient(phi_p, DIM(nx[n], ny[n], nz[n]));
    double norm = sqrt(SUMD(nx[n]*nx[n], ny[n]*ny[n], nz[n]*nz[n]));

    if(norm > EPS)
    {
      nx[n] /= norm;
      ny[n] /= norm;
#ifdef P4_TO_P8
      nz[n] /= norm;
#endif
    }
    else
    {
      nx[n] = 0.0;
      ny[n] = 0.0;
#ifdef P4_TO_P8
      nz[n] = 0.0;
#endif
    }
  }

  ierr = VecGetArray(q , &q_p) ; CHKERRXX(ierr);

  /* initialize qn */
  if(order >= 1)
  {
    ierr = VecDuplicate(phi, &qn); CHKERRXX(ierr);
    ierr = VecDuplicate(phi, &b_qn_well_defined ); CHKERRXX(ierr);

    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      ngbd->get_neighbors(n, qnnn);
      if(phi_p[qnnn.node_000] < -EPS &&
     #ifdef P4_TO_P8
           (phi_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (phi_p[qnnn.node_m00_mp] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           (phi_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (phi_p[qnnn.node_m00_pp] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           (phi_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (phi_p[qnnn.node_p00_mp] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           (phi_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (phi_p[qnnn.node_p00_pp] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           (phi_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (phi_p[qnnn.node_0m0_mp] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           (phi_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (phi_p[qnnn.node_0m0_pp] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           (phi_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (phi_p[qnnn.node_0p0_mp] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           (phi_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (phi_p[qnnn.node_0p0_pp] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           (phi_p[qnnn.node_00m_mm] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (phi_p[qnnn.node_00m_mp] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           (phi_p[qnnn.node_00m_pm] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (phi_p[qnnn.node_00m_pp] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           (phi_p[qnnn.node_00p_mm] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (phi_p[qnnn.node_00p_mp] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           (phi_p[qnnn.node_00p_pm] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (phi_p[qnnn.node_00p_pp] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           (phi_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS) &&
           (phi_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS) &&
           (phi_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS) &&
           (phi_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS) &&
           (phi_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS) &&
           (phi_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS) &&
           (phi_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS) &&
           (phi_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        double grad_q[P4EST_DIM]; qnnn.gradient(q_p, DIM(grad_q[0], grad_q[1], grad_q[2]));
        qn_p[n] = SUMD(nx[n]*grad_q[0], ny[n]*grad_q[1], nz[n]*grad_q[2]);
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0.0;
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

    for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000] == true &&
     #ifdef P4_TO_P8
           (b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_m00_mp] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_m00_pp] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           (b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_p00_mp] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_p00_pp] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           (b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0m0_mp] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0m0_pp] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           (b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0p0_mp] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0p0_pp] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           (b_qn_well_defined_p[qnnn.node_00m_mm] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_00m_mp] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_00m_pm] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_00m_pp] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           (b_qn_well_defined_p[qnnn.node_00p_mm] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_00p_mp] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_00p_pm] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_00p_pp] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           (b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS) &&
           (b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        double grad_qn[P4EST_DIM]; qnnn.gradient(qn_p, DIM(grad_qn[0], grad_qn[1], grad_qn[2]));
        qnn_p[n] = SUMD(nx[n]*grad_qn[0], ny[n]*grad_qn[1], nz[n]*grad_qn[2]);;
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0.0;
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
  if(order == 2)
  {
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
    for(int it = 0; it < iterations; ++it)
    {
      ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);

      for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
      {
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnnx = nx[n] > 0.0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                    : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n] > 0.0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                    : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n] > 0.0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                    : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif

          qnn_p[n] -= (dt*(SUMD(nx[n]*qnnx, ny[n]*qnny, nz[n]*qnnz)));
        }
      }
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }
    ierr = VecRestoreArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);
  }

  /* extrapolate qn */
  if(order >= 1)
  {
    if(order == 2) { ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr); }
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for(int it = 0; it < iterations; ++it)
    {
      ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);

      for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
      {
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnx = nx[n] > 0.0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                                   : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n] > 0.0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                                   : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n] > 0.0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                                   : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif

          qn_p[n] -= (dt*(SUMD(nx[n]*qnx, ny[n]*qny, nz[n]*qnz)) - (order == 2 ? dt*qnn_p[n] : 0.0));
        }
      }
      ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);

      ierr = VecGhostUpdateBegin(qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    }

    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
    if(order == 2) { ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr); }
  }
  ierr = VecRestoreArray(q, &q_p); CHKERRXX(ierr);

  if(order >= 1) { ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr); }
  if(order == 2) { ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr); }

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

  if(order >= 1) { ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr); }

  for(int it = 0; it < iterations; ++it)
  {
    ngbd->second_derivatives_central(q, DIM(qxx, qyy, qzz));

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);

    for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      if(phi_p[n] > -EPS)
      {
        ngbd->get_neighbors(n, qnnn);
        double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
        dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
        dt /= ((double) P4EST_DIM);

        /* first order one sided derivatives */
        double qx = nx[n] > 0.0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n] > 0.0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n] > 0.0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
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
        qxx_m00 = qxx_p[n]*qxx_m00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n] < 0.0) qx -= .5*qnnn.d_p00*qxx_p00;
        else            qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n] < 0.0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else            qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n] < 0.0) qz -= .5*qnnn.d_00p*qzz_00p;
        else            qz += .5*qnnn.d_00m*qzz_00m;
#endif

        q_p[n] -= (dt*(SUMD(nx[n]*qx, ny[n]*qy, nz[n]*qz)) - (order >= 1 ? dt*qn_p[n] : 0.0));
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
                                                                                    DIM(std::vector<double>& nx, std::vector<double>& ny, std::vector<double>& nz),
                                                                                    double *q_out_p,
                                                                                    double *q_p,
                                                                                    DIM(double *qxx_p, double *qyy_p, double *qzz_p),
                                                                                    std::vector<double>& qi_m00, std::vector<double>& qi_p00,
                                                                                    std::vector<double>& qi_0m0, std::vector<double>& qi_0p0
                                                                                    ONLY3D(COMMA std::vector<double>& qi_00m COMMA std::vector<double>& qi_00p),
                                                                                    std::vector<double>& s_m00 , std::vector<double>& s_p00,
                                                                                    std::vector<double>& s_0m0 , std::vector<double>& s_0p0
                                                                                    ONLY3D(COMMA std::vector<double>& s_00m COMMA std::vector<double>& s_00p)
                                                                                    ) const
{
  quad_neighbor_nodes_of_node_t qnnn;
  for(size_t n_map = 0; n_map < map.size(); ++n_map)
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
#endif
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0 ONLY3D(COMMA p_00m COMMA p_00p));
    qnnn.ngbd_with_quadratic_interpolation(q_p  , q_000, q_m00, q_p00, q_0m0, q_0p0 ONLY3D(COMMA q_00m COMMA q_00p));

    double s_p00_ = qnnn.d_p00; double s_m00_ = qnnn.d_m00;
    double s_0p0_ = qnnn.d_0p0; double s_0m0_ = qnnn.d_0m0;
#ifdef P4_TO_P8
    double s_00p_ = qnnn.d_00p; double s_00m_ = qnnn.d_00m;
#endif

    if(p_000*p_m00 < 0.0) {
      s_m00_ = s_m00[n];
      q_m00 = qi_m00[n];
    }
    if(p_000*p_p00 < 0.0) {
      s_p00_ = s_p00[n];
      q_p00 = qi_p00[n];
    }
    if(p_000*p_0m0 < 0.0) {
      s_0m0_ = s_0m0[n];
      q_0m0 = qi_0m0[n];
    }
    if(p_000*p_0p0 < 0.0){
      s_0p0_ = s_0p0[n];
      q_0p0 = qi_0p0[n];
    }
#ifdef P4_TO_P8
    if(p_000*p_00m < 0.0){
      s_00m_ = s_00m[n];
      q_00m = qi_00m[n];
    }
    if(p_000*p_00p < 0.0){
      s_00p_ = s_00p[n];
      q_00p = qi_00p[n];
    }
#endif

//    double sgn = (p_000 > 0.0) ? 1.0 : -1.0;
    double qxx_000, qxx_m00, qxx_p00, qxx_0m0, qxx_0p0;
    double qyy_000, qyy_m00, qyy_p00, qyy_0m0, qyy_0p0;
#ifdef P4_TO_P8
    double qxx_00m, qxx_00p;
    double qyy_00m, qyy_00p;
    double qzz_000, qzz_m00, qzz_p00, qzz_0m0, qzz_0p0, qzz_00m, qzz_00p;
#endif
    qnnn.ngbd_with_quadratic_interpolation(qxx_p, qxx_000, qxx_m00, qxx_p00, qxx_0m0, qxx_0p0 ONLY3D(COMMA qxx_00m COMMA qxx_00p));
    qnnn.ngbd_with_quadratic_interpolation(qyy_p, qyy_000, qyy_m00, qyy_p00, qyy_0m0, qyy_0p0 ONLY3D(COMMA qyy_00m COMMA qyy_00p));
#ifdef P4_TO_P8
    qnnn.ngbd_with_quadratic_interpolation(qzz_p, qzz_000, qzz_m00, qzz_p00, qzz_0m0, qzz_0p0, qzz_00m, qzz_00p);
#endif

    //---------------------------------------------------------------------
    // Neumann boundary condition on the walls
    //---------------------------------------------------------------------
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    /* wall in the x direction */
    if     (is_node_xmWall(p4est, node)) { s_m00_ = s_p00_; q_m00 = q_p00; qxx_000 = qxx_m00 = qxx_p00 = 0.0; }
    else if(is_node_xpWall(p4est, node)) { s_p00_ = s_m00_; q_p00 = q_m00; qxx_000 = qxx_m00 = qxx_p00 = 0.0; }

    /* wall in the y direction */
    if     (is_node_ymWall(p4est, node)) { s_0m0_ = s_0p0_; q_0m0 = q_0p0; qyy_000 = qyy_0m0 = qyy_0p0 = 0.0; }
    else if(is_node_ypWall(p4est, node)) { s_0p0_ = s_0m0_; q_0p0 = q_0m0; qyy_000 = qyy_0m0 = qyy_0p0 = 0.0; }

#ifdef P4_TO_P8
    /* wall in the y direction */
    if     (is_node_zmWall(p4est, node)) { s_00m_ = s_00p_; q_00m = q_00p; qzz_000 = qzz_00m = qzz_00p = 0.0; }
    else if(is_node_zpWall(p4est, node)) { s_00p_ = s_00m_; q_00p = q_00m; qzz_000 = qzz_00m = qzz_00p = 0.0; }
#endif

    //---------------------------------------------------------------------
    // Second order accurate One-Sided Differecing
    //---------------------------------------------------------------------
//    double qxm = (q_000- q_m00)/s_m00_ + 0.5*s_m00_*MC(qxx_m00, qxx_000);
//    double qxp = (q_p00- q_000)/s_p00_ - 0.5*s_p00_*MC(qxx_p00, qxx_000);
//    double qym = (q_000- q_0m0)/s_0m0_ + 0.5*s_0m0_*MC(qyy_0m0, qyy_000);
//    double qyp = (q_0p0- q_000)/s_0p0_ - 0.5*s_0p0_*MC(qyy_0p0, qyy_000);
//#ifdef P4_TO_P8
//    double qzm = (q_000- q_00m)/s_00m_ + 0.5*s_00m_*MC(qzz_00m, qzz_000);
//    double qzp = (q_00p- q_000)/s_00p_ - 0.5*s_00p_*MC(qzz_00p, qzz_000);
//#endif
    double qxm = (q_000- q_m00)/s_m00_ + 0.5*s_m00_*MINMOD(qxx_m00, qxx_000);
    double qxp = (q_p00- q_000)/s_p00_ - 0.5*s_p00_*MINMOD(qxx_p00, qxx_000);
    double qym = (q_000- q_0m0)/s_0m0_ + 0.5*s_0m0_*MINMOD(qyy_0m0, qyy_000);
    double qyp = (q_0p0- q_000)/s_0p0_ - 0.5*s_0p0_*MINMOD(qyy_0p0, qyy_000);
#ifdef P4_TO_P8
    double qzm = (q_000- q_00m)/s_00m_ + 0.5*s_00m_*MINMOD(qzz_00m, qzz_000);
    double qzp = (q_00p- q_000)/s_00p_ - 0.5*s_00p_*MINMOD(qzz_00p, qzz_000);
#endif

    //---------------------------------------------------------------------
    // Upwind Scheme
    //---------------------------------------------------------------------
    double dt = MIN(s_m00_, s_p00_);
    dt = MIN(dt, s_0m0_, s_0p0_);
#ifdef P4_TO_P8
    dt = MIN(dt, s_00m_, s_00p_);
#endif
    dt /= ((double) P4EST_DIM);

//    q_out_p[n] = q_000 - (dt*sgn) * ( nx[n]*( (sgn*nx[n] > 0.0) ? qxm : qxp) +
//                                      ny[n]*( (sgn*ny[n] > 0.0) ? qym : qyp)
//                                  #ifdef P4_TO_P8
//                                      + nz[n]*( (sgn*nz[n] > 0.0) ? qzm : qzp)
//                                  #endif
//                                      );
    q_out_p[n] = q_000 - dt*(SUMD(nx[n]*(nx[n] > 0.0 ? qxm : qxp), ny[n]*(ny[n] > 0.0 ? qym : qyp), nz[n]*(nz[n] > 0.0 ? qzm : qzp)));
  }
}

void my_p4est_level_set_t::extend_from_interface_to_whole_domain_TVD(Vec phi, Vec qi, Vec q, int iterations, Vec mask, double band_zero, double band_smooth, double (*cf)(p4est_locidx_t, int, double)) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_from_interface_TVD, phi, qi, q, 0); CHKERRXX(ierr);

  if (mask != NULL && cf != NULL) throw std::invalid_argument("No mask and cf simultaneously at the moment");

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
#endif
  double dl = MAX(DIM(dx, dy, dz));

  Vec qxxyyzz[P4EST_DIM];
  double *qxxyyzz_p[P4EST_DIM];
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecCreateGhostNodes(p4est, nodes, &qxxyyzz[dim]); CHKERRXX(ierr);
  }
  compute_derivatives(qi, qxxyyzz);

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

  for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);
    qnnn.gradient(phi_p, DIM(nx[n], ny[n], nz[n]));
    double norm = sqrt(SUMD(nx[n]*nx[n], ny[n]*ny[n], nz[n]*nz[n]));

    if(norm > EPS)
    {
      if (phi_p[n] < 0.0) norm = -norm;
      nx[n] /= norm;
      ny[n] /= norm;
#ifdef P4_TO_P8
      nz[n] /= norm;
#endif
    }
    else
    {
      nx[n] = 0.0;
      ny[n] = 0.0;
#ifdef P4_TO_P8
      nz[n] = 0.0;
#endif
    }
  }


  /* compute second order derivatives of phi for second order accurate location */
  Vec dxxyyzz[P4EST_DIM]; double *dxxyyzz_p[P4EST_DIM];
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecCreateGhostNodes(p4est, nodes, &dxxyyzz[dim]); CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  compute_derivatives(phi, dxxyyzz);

  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecGetArray(dxxyyzz[dim], &dxxyyzz_p[dim]); CHKERRXX(ierr);
  }

  /* initialization of q */
  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  if (cf == NULL)
  {
    if (mask == NULL)
    {
      for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
        q_p[n] = fabs(phi_p[n])<1.5*dl ? qi_p[n] : 0;
    } else {
      double *mask_p;
      ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);
      for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
        q_p[n] = MAX(mask_p[n], fabs(phi_p[n]))<1.5*dl ? qi_p[n] : 0;
      ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
    }
  } else {
    for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
      q_p[n] = 0.0;
  }

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
#endif

  my_p4est_interpolation_nodes_t interp_m00(ngbd); interp_m00.set_input(qi, DIM(qxxyyzz[0], qxxyyzz[1], qxxyyzz[2]), interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_p00(ngbd); interp_p00.set_input(qi, DIM(qxxyyzz[0], qxxyyzz[1], qxxyyzz[2]), interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_0m0(ngbd); interp_0m0.set_input(qi, DIM(qxxyyzz[0], qxxyyzz[1], qxxyyzz[2]), interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_0p0(ngbd); interp_0p0.set_input(qi, DIM(qxxyyzz[0], qxxyyzz[1], qxxyyzz[2]), interpolation_on_interface);
#ifdef P4_TO_P8
  my_p4est_interpolation_nodes_t interp_00m(ngbd); interp_00m.set_input(qi, DIM(qxxyyzz[0], qxxyyzz[1], qxxyyzz[2]), interpolation_on_interface);
  my_p4est_interpolation_nodes_t interp_00p(ngbd); interp_00p.set_input(qi, DIM(qxxyyzz[0], qxxyyzz[1], qxxyyzz[2]), interpolation_on_interface);
#endif

  my_p4est_interpolation_nodes_t interp_mask(ngbd); interp_mask.set_input(mask, linear);

  for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
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
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0 ONLY3D(COMMA p_00m COMMA p_00p));

    double s_p00_ = qnnn.d_p00; double s_m00_ = qnnn.d_m00;
    double s_0p0_ = qnnn.d_0p0; double s_0m0_ = qnnn.d_0m0;
#ifdef P4_TO_P8
    double s_00p_ = qnnn.d_00p; double s_00m_ = qnnn.d_00m;
#endif

    //---------------------------------------------------------------------
    // Second Order derivatives
    //---------------------------------------------------------------------
    double pxx_000 = dxxyyzz_p[0][n];
    double pyy_000 = dxxyyzz_p[1][n];
#ifdef P4_TO_P8
    double pzz_000 = dxxyyzz_p[2][n];
#endif
    double pxx_m00 = qnnn.f_m00_linear(dxxyyzz_p[0]);
    double pxx_p00 = qnnn.f_p00_linear(dxxyyzz_p[0]);
    double pyy_0m0 = qnnn.f_0m0_linear(dxxyyzz_p[1]);
    double pyy_0p0 = qnnn.f_0p0_linear(dxxyyzz_p[1]);
#ifdef P4_TO_P8
    double pzz_00m = qnnn.f_00m_linear(dxxyyzz_p[2]);
    double pzz_00p = qnnn.f_00p_linear(dxxyyzz_p[2]);
#endif
    double dist;

    if(p_000*p_m00 < 0.0)
    {
//      dist = interface_Location(0, s_m00_, p_000, p_m00);
      dist = interface_Location_With_Second_Order_Derivative(0, s_m00_, p_000, p_m00, pxx_000, pxx_m00);
      dist = MAX(dist, EPS);
      double xyz[] = { DIM(x - dist, y, z)};
      if (cf == NULL) interp_m00.add_point(n, xyz);
      else qi_m00[n] = (*cf)(n, dir::f_m00, dist);
      s_m00[n] = dist;
    }
    else
    {
      qi_m00[n] = qi_p[n];
      s_m00[n] = s_m00_;
    }

    if(p_000*p_p00 < 0.0)
    {
//      dist = interface_Location(0, s_p00_, p_000, p_p00);
      dist = interface_Location_With_Second_Order_Derivative(0, s_p00_, p_000, p_p00, pxx_000, pxx_p00);
      dist = MAX(dist, EPS);
      double xyz[] = { DIM(x + dist, y, z)};
        if (cf == NULL) interp_p00.add_point(n, xyz);
        else qi_p00[n] = (*cf)(n, dir::f_p00, dist);
        s_p00[n] = dist;
    }
    else
    {
      qi_p00[n] = qi_p[n];
      s_p00[n] = s_p00_;
    }

    if(p_000*p_0m0 < 0.0) {
//      dist = interface_Location(0, s_0m0_, p_000, p_0m0);
      dist = interface_Location_With_Second_Order_Derivative(0, s_0m0_, p_000, p_0m0, pyy_000, pyy_0m0);
      dist = MAX(dist, EPS);
      double xyz[] = { DIM(x, y - dist, z)};
        if (cf == NULL) interp_0m0.add_point(n, xyz);
        else qi_0m0[n] = (*cf)(n, dir::f_0m0, dist);
        s_0m0[n] = dist;
    }
    else
    {
      qi_0m0[n] = qi_p[n];
      s_0m0[n] = s_0m0_;
    }

    if(p_000*p_0p0 < 0.0)
    {
//      dist = interface_Location(0, s_0p0_, p_000, p_0p0);
      dist = interface_Location_With_Second_Order_Derivative(0, s_0p0_, p_000, p_0p0, pyy_000, pyy_0p0);
      dist = MAX(dist, EPS);
      double xyz[] = { DIM(x, y + dist, z)};

        if (cf == NULL) interp_0p0.add_point(n, xyz);
        else qi_0p0[n] = (*cf)(n, dir::f_0p0, dist);
        s_0p0[n] = dist;
    }
    else
    {
      qi_0p0[n] = qi_p[n];
      s_0p0[n] = s_0p0_;
    }

#ifdef P4_TO_P8
    if(p_000*p_00m < 0.0)
    {
//      dist = interface_Location(0, s_00m_, p_000, p_00m);
      dist = interface_Location_With_Second_Order_Derivative(0, s_00m_, p_000, p_00m, pzz_000, pzz_00m);
      dist = MAX(dist, EPS);
      double xyz[] = { x, y, z - dist };

        if (cf == NULL) interp_00m.add_point(n, xyz);
        else qi_00m[n] = (*cf)(n, dir::f_00m, dist);
        s_00m[n] = dist;
    }
    else
    {
      qi_00m[n] = qi_p[n];
      s_00m[n] = s_00m_;
    }

    if(p_000*p_00p < 0.0)
    {
//      dist = interface_Location(0, s_00p_, p_000, p_00p);
      dist = interface_Location_With_Second_Order_Derivative(0, s_00p_, p_000, p_00p, pzz_000, pzz_00p);
      dist = MAX(dist, EPS);
      double xyz[] = { x, y, z + dist };

        if (cf == NULL) interp_00p.add_point(n, xyz);
        else qi_00p[n] = (*cf)(n, dir::f_00p, dist);
        s_00p[n] = dist;
    }
    else
    {
      qi_00p[n] = qi_p[n];
      s_00p[n] = s_00p_;
    }
#endif
  }

  if (cf == NULL)
  {
    interp_m00.interpolate(qi_m00.data());
    interp_p00.interpolate(qi_p00.data());
    interp_0m0.interpolate(qi_0m0.data());
    interp_0p0.interpolate(qi_0p0.data());
#ifdef P4_TO_P8
    interp_00m.interpolate(qi_00m.data());
    interp_00p.interpolate(qi_00p.data());
#endif
  }

  if (mask != NULL)
  {
    std::vector<double> mask_m00(nodes->num_owned_indeps, 0); interp_m00.set_input(mask, linear); interp_m00.interpolate(mask_m00.data());
    std::vector<double> mask_p00(nodes->num_owned_indeps, 0); interp_p00.set_input(mask, linear); interp_p00.interpolate(mask_p00.data());

    std::vector<double> mask_0m0(nodes->num_owned_indeps, 0); interp_0m0.set_input(mask, linear); interp_0m0.interpolate(mask_0m0.data());
    std::vector<double> mask_0p0(nodes->num_owned_indeps, 0); interp_0p0.set_input(mask, linear); interp_0p0.interpolate(mask_0p0.data());
#ifdef P4_TO_P8
    std::vector<double> mask_00m(nodes->num_owned_indeps, 0); interp_00m.set_input(mask, linear); interp_00m.interpolate(mask_00m.data());
    std::vector<double> mask_00p(nodes->num_owned_indeps, 0); interp_00p.set_input(mask, linear); interp_00p.interpolate(mask_00p.data());
#endif

    double a = band_zero*dl;
    double b = band_smooth*dl;

    foreach_local_node(n, nodes)
    {
      ngbd->get_neighbors(n, qnnn);

      double p_000, p_m00, p_p00, p_0m0, p_0p0;
#ifdef P4_TO_P8
      double p_00m, p_00p;
#endif
      qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0 ONLY3D(COMMA p_00m COMMA p_00p));

      if (p_000*p_m00 < 0.0) { qi_m00[n] *= 1. - smoothstep(1, (mask_m00[n] - a)/b); }
      if (p_000*p_p00 < 0.0) { qi_p00[n] *= 1. - smoothstep(1, (mask_p00[n] - a)/b); }

      if (p_000*p_0m0 < 0.0) { qi_0m0[n] *= 1. - smoothstep(1, (mask_0m0[n] - a)/b); }
      if (p_000*p_0p0 < 0.0) { qi_0p0[n] *= 1. - smoothstep(1, (mask_0p0[n] - a)/b); }
#ifdef P4_TO_P8
      if (p_000*p_00m < 0.0) { qi_00m[n] *= 1. - smoothstep(1, (mask_00m[n] - a)/b); }
      if (p_000*p_00p < 0.0) { qi_00p[n] *= 1. - smoothstep(1, (mask_00p[n] - a)/b); }
#endif
    }
  }

  for(int it = 0; it < iterations; ++it)
  {
    //---------------------------------------------------------------------
    // q1 = q - dt*sgn(phi)*n \cdot \nabla(q) by the Godunov scheme with ENO-2 and subcell resolution
    //---------------------------------------------------------------------

    compute_derivatives(q, qxxyyzz);
    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecGetArray(qxxyyzz[dim], &qxxyyzz_p[dim]); CHKERRXX(ierr);
    }

    /* First do layer nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration(layer_nodes, phi_p, DIM(nx, ny, nz),
                                                            q1_p,
                                                            q_p, DIM(qxxyyzz_p[0], qxxyyzz_p[1], qxxyyzz_p[2]),
                                                            qi_m00, qi_p00,
                                                            qi_0m0, qi_0p0
                                                            ONLY3D(COMMA qi_00m COMMA qi_00p),
                                                            s_m00, s_p00,
                                                            s_0m0, s_0p0
                                                            ONLY3D(COMMA s_00m COMMA s_00p));

    /* initiate communication for q1 */
    ierr = VecGhostUpdateBegin(q1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* compute the local nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration(local_nodes, phi_p, DIM(nx, ny, nz),
                                                            q1_p,
                                                            q_p, DIM(qxxyyzz_p[0], qxxyyzz_p[1], qxxyyzz_p[2]),
                                                            qi_m00, qi_p00,
                                                            qi_0m0, qi_0p0
                                                            ONLY3D(COMMA qi_00m COMMA qi_00p),
                                                            s_m00, s_p00,
                                                            s_0m0, s_0p0
                                                            ONLY3D(COMMA s_00m COMMA s_00p));

    /* finish communication for q1 */
    ierr = VecGhostUpdateEnd(q1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArray(qxxyyzz[dim], &qxxyyzz_p[dim]); CHKERRXX(ierr);
    }

    compute_derivatives(q, qxxyyzz);

    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecGetArray(qxxyyzz[dim], &qxxyyzz_p[dim]); CHKERRXX(ierr);
    }

    //---------------------------------------------------------------------
    // q2 = q1 - dt*sgn(phi)*n \cdot \nabla(q1) by the Godunov scheme with ENO-2 and subcell resolution
    //---------------------------------------------------------------------
    /* First do layer nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration(layer_nodes, phi_p, DIM(nx, ny, nz),
                                                            q2_p,
                                                            q1_p, DIM(qxxyyzz_p[0], qxxyyzz_p[1], qxxyyzz_p[2]),
                                                            qi_m00, qi_p00,
                                                            qi_0m0, qi_0p0
                                                            ONLY3D(COMMA qi_00m COMMA qi_00p),
                                                            s_m00, s_p00,
                                                            s_0m0, s_0p0
                                                            ONLY3D(COMMA s_00m COMMA s_00p));

    /* initiate communication for q2 */
    ierr = VecGhostUpdateBegin(q2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* compute the local nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration(local_nodes, phi_p, DIM(nx, ny, nz),
                                                            q2_p,
                                                            q1_p, DIM(qxxyyzz_p[0], qxxyyzz_p[1], qxxyyzz_p[2]),
                                                            qi_m00, qi_p00,
                                                            qi_0m0, qi_0p0
                                                            ONLY3D(COMMA qi_00m COMMA qi_00p),
                                                            s_m00, s_p00,
                                                            s_0m0, s_0p0
                                                            ONLY3D(COMMA s_00m COMMA s_00p));

    /* finish communication for q2 */
    ierr = VecGhostUpdateEnd(q2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
      ierr = VecRestoreArray(qxxyyzz[dim], &qxxyyzz_p[dim]); CHKERRXX(ierr);
    }

    if (show_convergence)
    {
      double change = 0.0;
      for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
      {
        if (fabs(phi_p[n]) < show_convergence_band*dl)
        {
          double change_loc = 0.5*fabs(q2_p[n] - q_p[n]);
          if (change_loc > change) change = change_loc;
        }
      }

      int mpiret = MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm); SC_CHECK_MPI(mpiret);

      ierr = PetscPrintf(p4est->mpicomm, "Extending in normal direction, iteration: %d, error: %e\n", it, change); CHKERRXX(ierr);
    }

    //---------------------------------------------------------------------
    // The third step of TVD RK-2 : q = .5*(q + q2)
    //---------------------------------------------------------------------
    for(size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
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
  for (unsigned char dim = 0; dim < P4EST_DIM; ++dim) {
    ierr = VecDestroy(qxxyyzz[dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(dxxyyzz[dim], &dxxyyzz_p[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(dxxyyzz[dim]);  CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_my_p4est_level_set_extend_from_interface_TVD, phi, qi, q, 0); CHKERRXX(ierr);
}

void my_p4est_level_set_t::extend_from_interface_to_whole_domain_TVD_one_iteration_daniil_ver(const std::vector<int>& map, DIM(vector<double>& nx, vector<double>& ny, vector<double>& nz),
                                                                                   double *q_out_p, double *q_p, DIM(double *qxx_p, double *qyy_p, double *qzz_p),
                                                                                   std::vector<int>& map_grid_to_interface,
                                                                                   std::vector<int>& interface_directions,
                                                                                   std::vector<double>& interface_distances,
                                                                                   std::vector<double>& interface_values) const
{
  for (size_t n_map = 0; n_map < map.size(); ++n_map) {
    p4est_locidx_t n = map[n_map];

    const quad_neighbor_nodes_of_node_t qnnn = (*ngbd)[n];

    //---------------------------------------------------------------------
    // Neighborhood information
    //---------------------------------------------------------------------
    double q_000, q_nei[P4EST_FACES];
    qnnn.ngbd_with_quadratic_interpolation(q_p, q_000, q_nei);

    double s_nei_[P4EST_FACES] = { DIMPM(qnnn.d_m00, qnnn.d_p00,
                                         qnnn.d_0m0, qnnn.d_0p0,
                                         qnnn.d_00m, qnnn.d_00p) };
    //---------------------------------------------------------------------
    // Fetch interface values
    //---------------------------------------------------------------------
    for (int i = map_grid_to_interface[n]; i < map_grid_to_interface[n+1]; ++i) {
      int dir     = interface_directions[i];
      s_nei_[dir] = interface_distances[i];
      q_nei [dir] = interface_values[i];
    }

    //---------------------------------------------------------------------
    // Fetch second derivatives and determine direction
    //---------------------------------------------------------------------
    double DIM(qxx, qyy, qzz);
    int    DIM(nei_x, nei_y, nei_z);

    if (nx[n] > 0) { nei_x = dir::f_m00; qxx = MINMOD(qxx_p[n], qnnn.f_m00_linear(qxx_p));}
    else           { nei_x = dir::f_p00; qxx = MINMOD(qxx_p[n], qnnn.f_p00_linear(qxx_p));}

    if (ny[n] > 0) { nei_y = dir::f_0m0; qyy = MINMOD(qyy_p[n], qnnn.f_0m0_linear(qyy_p));}
    else           { nei_y = dir::f_0p0; qyy = MINMOD(qyy_p[n], qnnn.f_0p0_linear(qyy_p));}
#ifdef P4_TO_P8
    if (nz[n] > 0) { nei_z = dir::f_00m; qzz = MINMOD(qzz_p[n], qnnn.f_00m_linear(qzz_p));}
    else           { nei_z = dir::f_00p; qzz = MINMOD(qzz_p[n], qnnn.f_00p_linear(qzz_p));}
#endif

    //---------------------------------------------------------------------
    // Neumann boundary condition on the walls
    //---------------------------------------------------------------------
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    /* wall in the x direction */
    if     (is_node_xmWall(p4est, node)) { s_nei_[dir::f_m00] = s_nei_[dir::f_p00]; q_nei[dir::f_m00] = q_nei[dir::f_p00]; qxx = 0; }
    else if(is_node_xpWall(p4est, node)) { s_nei_[dir::f_p00] = s_nei_[dir::f_m00]; q_nei[dir::f_p00] = q_nei[dir::f_m00]; qxx = 0; }

    /* wall in the y direction */
    if     (is_node_ymWall(p4est, node)) { s_nei_[dir::f_0m0] = s_nei_[dir::f_0p0]; q_nei[dir::f_0m0] = q_nei[dir::f_0p0]; qyy = 0; }
    else if(is_node_ypWall(p4est, node)) { s_nei_[dir::f_0p0] = s_nei_[dir::f_0m0]; q_nei[dir::f_0p0] = q_nei[dir::f_0m0]; qyy = 0; }
#ifdef P4_TO_P8
    /* wall in the y direction */
    if     (is_node_zmWall(p4est, node)) { s_nei_[dir::f_00m] = s_nei_[dir::f_00p]; q_nei[dir::f_00m] = q_nei[dir::f_00p]; qzz = 0; }
    else if(is_node_zpWall(p4est, node)) { s_nei_[dir::f_00p] = s_nei_[dir::f_00m]; q_nei[dir::f_00p] = q_nei[dir::f_00m]; qzz = 0; }
#endif

    //---------------------------------------------------------------------
    // Second order accurate One-Sided Differecing
    //---------------------------------------------------------------------
    double DIM( qxnx = fabs(nx[n])*((q_000-q_nei[nei_x])/s_nei_[nei_x] + 0.5*s_nei_[nei_x]*qxx),
                qyny = fabs(ny[n])*((q_000-q_nei[nei_y])/s_nei_[nei_y] + 0.5*s_nei_[nei_y]*qyy),
                qznz = fabs(nz[n])*((q_000-q_nei[nei_z])/s_nei_[nei_z] + 0.5*s_nei_[nei_z]*qzz) );

    //---------------------------------------------------------------------
    // Upwind Scheme
    //---------------------------------------------------------------------
    double dt = MIN(MIN(DIM(s_nei_[dir::f_m00], s_nei_[dir::f_0m0], s_nei_[dir::f_00m])),
                    MIN(DIM(s_nei_[dir::f_p00], s_nei_[dir::f_0p0], s_nei_[dir::f_00p])))/double(P4EST_DIM);

    q_out_p[n] = q_000 - dt * SUMD(qxnx, qyny, qznz);
  }
}

void my_p4est_level_set_t::extend_from_interface_to_whole_domain_TVD_daniil_ver(Vec phi, Vec qi, Vec q, int iterations,
                                                                     Vec mask, double band_zero, double band_smooth,
                                                                     double (*func)(p4est_locidx_t, int, double)) const
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_my_p4est_level_set_extend_from_interface_TVD, phi, qi, q, 0); CHKERRXX(ierr);

  if (qi == NULL && func == NULL) {
    throw std::invalid_argument("Must provide at least one: qi or func");
  }

  /* init the neighborhood information if needed */
  /* NOTE: from now on the neighbors will be initialized ... do we want to clear them
   * at the end of this function if they were not initialized beforehand ?
   */
  ngbd->init_neighbors();

  /* find dx and dy smallest */
  double dxyz[P4EST_DIM];
  p4est_dxyz_min(p4est, dxyz);
  double dl = ABSD(dxyz[0], dxyz[1], dxyz[2]);

  /* compute second order derivatives of phi and q*/
  // note that even if we use func to compute interface values,
  // these allocated q_dd's will still be used during iterations
  Vec     phi_dd  [P4EST_DIM];
  double *phi_dd_p[P4EST_DIM];

  Vec     q_dd  [P4EST_DIM];
  double *q_dd_p[P4EST_DIM];

  foreach_dimension(dim) {
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_dd[dim]); CHKERRXX(ierr);
    ierr = VecDuplicate(phi_dd[dim], &q_dd[dim]); CHKERRXX(ierr);
  }

  compute_derivatives(phi, DIM(phi_dd[0], phi_dd[1], phi_dd[2]));
  if (func == NULL) {
    compute_derivatives(qi, DIM(q_dd[0], q_dd[1], q_dd[2]));
  }

  double *q_p, *qi_p, *phi_p, *mask_p;

  ierr = VecGetArray(q,   &q_p);   CHKERRXX(ierr);
  ierr = VecGetArray(qi,  &qi_p);  CHKERRXX(ierr);
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  if (mask != NULL) {
    band_zero   *= dl;
    band_smooth *= dl;
    ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);
  }

  foreach_dimension(dim) {
    ierr = VecGetArray(phi_dd[dim], &phi_dd_p[dim]); CHKERRXX(ierr);
    ierr = VecGetArray(q_dd  [dim], &q_dd_p  [dim]); CHKERRXX(ierr);
  }

  /* compute the normals */
  std::vector<double> DIM(nx(nodes->num_owned_indeps),
                          ny(nodes->num_owned_indeps),
                          nz(nodes->num_owned_indeps));

  foreach_local_node(n, nodes) {
    const quad_neighbor_nodes_of_node_t &qnnn = (*ngbd)[n];
    EXECD(nx[n] = qnnn.dx_central(phi_p),
          ny[n] = qnnn.dy_central(phi_p),
          nz[n] = qnnn.dz_central(phi_p));
    double norm = ABSD(nx[n], ny[n], nz[n]);

    if(norm > EPS) {
      if (phi_p[n] < 0) norm = -norm;
      EXECD(nx[n] /= norm,
            ny[n] /= norm,
            nz[n] /= norm);
    } else {
      EXECD(nx[n] = 0,
            ny[n] = 0,
            nz[n] = 0);
    }
  }

  /* initialize the quantities at the interface (instead of doing it each time in the loop ...) */
  std::vector<int>    map_grid_to_interface(nodes->num_owned_indeps+1); // map_grid_to_interface[n]: starting position, map_grid_to_interface[n+1]-map_grid_to_interface[n]: number of elements
  std::vector<int>    interface_directions;
  std::vector<double> interface_values;
  std::vector<double> interface_distances;

  map_grid_to_interface[0] = 0;
  foreach_local_node(n, nodes)
  {
    const quad_neighbor_nodes_of_node_t &qnnn = (*ngbd)[n];

    double xyz_n[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz_n);

    double p_000, p_nei[P4EST_FACES];
    qnnn.ngbd_with_quadratic_interpolation(phi_p, p_000, p_nei);

    double s_nei_[P4EST_FACES] = { DIMPM(qnnn.d_p00, qnnn.d_m00,
                                         qnnn.d_0p0, qnnn.d_0m0,
                                         qnnn.d_00p, qnnn.d_00m) };

    //---------------------------------------------------------------------
    // Second Order derivatives
    //---------------------------------------------------------------------
    double q_000, q_nei[P4EST_FACES];
    bool   q_ngbd_initialized = false;

    double mask_000, mask_nei[P4EST_FACES];
    bool   mask_ngbd_initialized = false;

    map_grid_to_interface[n+1] = map_grid_to_interface[n];

    foreach_direction(dir) {
      int dim = dir/2;
      if (p_000*p_nei[dir] < 0) {

        // find intersection
        double dist = interface_Location_With_Second_Order_Derivative(0, s_nei_[dir], p_000, p_nei[dir], phi_dd_p[dim][n], qnnn.f_linear(dir, phi_dd_p[dim]));
        dist = MAX(dist,EPS);

        double value;
        // compute value at intersection
        if (func == NULL) {
          if (!q_ngbd_initialized) {
            qnnn.ngbd_with_quadratic_interpolation(qi_p, q_000, q_nei);
            q_ngbd_initialized = true;
          }
          double q_dd_minmod = MINMOD(q_dd_p[dim][n], qnnn.f_linear(dir, q_dd_p[dim]));
          value = q_000*(1-dist/s_nei_[dir]) + q_nei[dir]*dist/s_nei_[dir] + 0.5*dist*(dist-s_nei_[dir])*q_dd_minmod;
        } else {
          value = func(n, dir, dist);
        }

        // fade to zero according to mask if needed
        if (mask != NULL) {
          if (!mask_ngbd_initialized) {
            qnnn.ngbd_with_quadratic_interpolation(mask_p, mask_000, mask_nei);
            mask_ngbd_initialized = true;
          }
          double mask_val = mask_000*(1-dist/s_nei_[dir]) + mask_nei[dir]*dist/s_nei_[dir];
          value *= 1.-smoothstep(1, (mask_val-band_zero)/band_smooth);
        }

        // save info
        map_grid_to_interface[n+1]++;
        interface_distances.push_back(dist);
        interface_directions.push_back(dir);
        interface_values.push_back(value);
      }
    }
  }

  /* initialization of q near interface (for faster convergence) */
  if (qi != NULL) { // for simplicity we will provide initial guess only through qi
    if (mask == NULL) {
      foreach_node(n, nodes) {
        q_p[n] = fabs(phi_p[n]) < 1.5*dl ? qi_p[n] : 0;
      }
    } else {
      double *mask_p;
      ierr = VecGetArray(mask, &mask_p); CHKERRXX(ierr);
      foreach_node(n, nodes) {
        q_p[n] = MAX(mask_p[n], fabs(phi_p[n])) < 1.5*dl ? qi_p[n] : 0;
      }
      ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
    }
  } else {
    foreach_node(n, nodes) {
      q_p[n] = 0;
    }
  }

  // Auxiliary arrays for time-stepping
  Vec     q1,    q2;
  double *q1_p, *q2_p;
  ierr = VecDuplicate(phi, &q1); CHKERRXX(ierr);
  ierr = VecDuplicate(phi, &q2); CHKERRXX(ierr);
  ierr = VecGetArray(q1, &q1_p); CHKERRXX(ierr);
  ierr = VecGetArray(q2, &q2_p); CHKERRXX(ierr);

  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  for (int it = 0; it < iterations; ++it)
  {
    //---------------------------------------------------------------------
    // q1 = q - dt*sgn(phi)*n \cdot \nabla(q) by the Godunov scheme with ENO-2 and subcell resolution
    //---------------------------------------------------------------------
    foreach_dimension(dim) { ierr = VecRestoreArray(q_dd[dim], &q_dd_p[dim]); CHKERRXX(ierr); }
    compute_derivatives(q1, DIM(q_dd[0], q_dd[1], q_dd[2]));
    foreach_dimension(dim) { ierr = VecGetArray(q_dd[dim], &q_dd_p[dim]); CHKERRXX(ierr);}

    /* First do layer nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration_daniil_ver(layer_nodes, DIM(nx, ny, nz),
        q1_p, q_p, DIM(q_dd_p[0], q_dd_p[1], q_dd_p[2]),
        map_grid_to_interface, interface_directions, interface_distances, interface_values);

    /* initiate communication for q1 */
    ierr = VecGhostUpdateBegin(q1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* compute the local nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration_daniil_ver(local_nodes, DIM(nx, ny, nz),
        q1_p, q_p, DIM(q_dd_p[0], q_dd_p[1], q_dd_p[2]),
        map_grid_to_interface, interface_directions, interface_distances, interface_values);

    /* finish communication for q1 */
    ierr = VecGhostUpdateEnd(q1, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    foreach_dimension(dim) { ierr = VecRestoreArray(q_dd[dim], &q_dd_p[dim]); CHKERRXX(ierr); }
    compute_derivatives(q1, DIM(q_dd[0], q_dd[1], q_dd[2]));
    foreach_dimension(dim) { ierr = VecGetArray(q_dd[dim], &q_dd_p[dim]); CHKERRXX(ierr);}

    //---------------------------------------------------------------------
    // q2 = q1 - dt*sgn(phi)*n \cdot \nabla(q1) by the Godunov scheme with ENO-2 and subcell resolution
    //---------------------------------------------------------------------
    /* First do layer nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration_daniil_ver(layer_nodes, DIM(nx, ny, nz),
        q2_p, q1_p, DIM(q_dd_p[0], q_dd_p[1], q_dd_p[2]),
        map_grid_to_interface, interface_directions, interface_distances, interface_values);

    /* initiate communication for q2 */
    ierr = VecGhostUpdateBegin(q2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* compute the local nodes */
    extend_from_interface_to_whole_domain_TVD_one_iteration_daniil_ver(local_nodes, DIM(nx, ny, nz),
        q2_p, q1_p, DIM(q_dd_p[0], q_dd_p[1], q_dd_p[2]),
        map_grid_to_interface, interface_directions, interface_distances, interface_values);

    /* finish communication for q2 */
    ierr = VecGhostUpdateEnd(q2, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    //---------------------------------------------------------------------
    // The third step of TVD RK-2 : q = .5*(q + q2)
    //---------------------------------------------------------------------
    foreach_node(n, nodes) {
      q_p[n] = .5*(q_p[n] + q2_p[n]);
    }
  }

  /* destroy the local petsc vectors */
  ierr = VecRestoreArray(q1, &q1_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(q2, &q2_p); CHKERRXX(ierr);
  ierr = VecDestroy(q1); CHKERRXX(ierr);
  ierr = VecDestroy(q2); CHKERRXX(ierr);

  foreach_dimension(dim) {
    ierr = VecRestoreArray(q_dd  [dim], &q_dd_p  [dim]); CHKERRXX(ierr);
    ierr = VecRestoreArray(phi_dd[dim], &phi_dd_p[dim]); CHKERRXX(ierr);
  }

  if (mask != NULL) {
    ierr = VecRestoreArray(mask, &mask_p); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(qi,  &qi_p);  CHKERRXX(ierr);
  ierr = VecRestoreArray(q,   &q_p);   CHKERRXX(ierr);

  foreach_dimension(dim) {
    ierr = VecDestroy(q_dd[dim]); CHKERRXX(ierr);
    ierr = VecDestroy(phi_dd[dim]); CHKERRXX(ierr);
  }

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
    for (short dim = 0; dim < P4EST_DIM; ++dim){
      ierr = VecGetArray(normal[dim], &normal_p[dim]); CHKERRXX(ierr);
    }

  std::vector<double> nx(nodes->num_owned_indeps);
  std::vector<double> ny(nodes->num_owned_indeps);
#ifdef P4_TO_P8
  std::vector<double> nz(nodes->num_owned_indeps);
#endif

  quad_neighbor_nodes_of_node_t qnnn;
  if (normal != NULL) {
    for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      nx[n] = normal_p[0][n];
      ny[n] = normal_p[1][n];
#ifdef P4_TO_P8
      nz[n] = normal_p[2][n];
#endif
    }
  } else {
    for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      ngbd->get_neighbors(n, qnnn);
      qnnn.gradient(phi_wall_p, DIM(nx[n], ny[n], nz[n]));
      double norm = sqrt(SUMD(nx[n]*nx[n], ny[n]*ny[n], nz[n]*nz[n]));

      if(norm > EPS)
      {
        nx[n] /= norm;
        ny[n] /= norm;
#ifdef P4_TO_P8
        nz[n] /= norm;
#endif
      }
      else
      {
        nx[n] = 0.0;
        ny[n] = 0.0;
#ifdef P4_TO_P8
        nz[n] = 0.0;
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

  for(int it = 0; it < iterations; ++it)
  {
    ngbd->second_derivatives_central(phi_intf, DIM(qxx, qyy, qzz));

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(phi_intf, &q_p); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_wall_p[n] > EPS)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
        dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
        dt /= ((double) P4EST_DIM);
        dt *= 0.45;

        /* first order one sided derivatives */
        double qx = nx[n] > 0.0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n] > 0.0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n] > 0.0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
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
        qxx_m00 = qxx_p[n]*qxx_m00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n] < 0.0) qx -= .5*qnnn.d_p00*qxx_p00;
        else            qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n] < 0.0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else            qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n] < 0.0) qz -= .5*qnnn.d_00p*qzz_00p;
        else            qz += .5*qnnn.d_00m*qzz_00m;
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

//        if(qx_p00 > 0.0) qx_p00 = 0.0;
//        if(qx_m00 < 0.0) qx_m00 = 0.0;
//        if(qy_0p0 > 0.0) qy_0p0 = 0.0;
//        if(qy_0m0 < 0.0) qy_0m0 = 0.0;
//#ifdef P4_TO_P8
//        if(qz_00p > 0.0) qz_00p = 0.0;
//        if(qz_00m < 0.0) qz_00m = 0.0;
//#endif
//        if(qx_p00 < 0.0) qx_p00 = 0.0;
//        if(qx_m00 > 0.0) qx_m00 = 0.0;
//        if(qy_0p0 < 0.0) qy_0p0 = 0.0;
//        if(qy_0m0 > 0.0) qy_0m0 = 0.0;
//#ifdef P4_TO_P8
//        if(qz_00p < 0.0) qz_00p = 0.0;
//        if(qz_00m > 0.0) qz_00m = 0.0;
//#endif

        double dot_product = SUMD( nx[n]*qx,ny[n]*qy,nz[n]*qz );

        if(cos_angle_p[n] < 0.0) {
          if(qx_p00 > 0.0) qx_p00 = 0.0;
          if(qx_m00 < 0.0) qx_m00 = 0.0;
          if(qy_0p0 > 0.0) qy_0p0 = 0.0;
          if(qy_0m0 < 0.0) qy_0m0 = 0.0;
  #ifdef P4_TO_P8
          if(qz_00p > 0.0) qz_00p = 0.0;
          if(qz_00m < 0.0) qz_00m = 0.0;
  #endif
        } else {
          if(qx_p00 < 0.0) qx_p00 = 0.0;
          if(qx_m00 > 0.0) qx_m00 = 0.0;
          if(qy_0p0 < 0.0) qy_0p0 = 0.0;
          if(qy_0m0 > 0.0) qy_0m0 = 0.0;
  #ifdef P4_TO_P8
          if(qz_00p < 0.0) qz_00p = 0.0;
          if(qz_00m > 0.0) qz_00m = 0.0;
  #endif
        }

        double abs_grad_q = sqrt( SUMD( MAX( qx_p00*qx_p00, qx_m00*qx_m00 ),
                                        MAX( qy_0p0*qy_0p0, qy_0m0*qy_0m0 ),
                                        MAX( qz_00p*qz_00p, qz_00m*qz_00m ) ) );

        tmp_p[n] = q_p[n] - dt*(dot_product - cos_angle_p[n]*abs_grad_q);
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_wall_p[n] > -EPS)
      {
        double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
        dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
        dt /= ((double) P4EST_DIM);
        dt *= 0.45;

        /* first order one sided derivatives */
        double qx = nx[n] > 0.0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n] > 0.0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n] > 0.0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
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
        qxx_m00 = qxx_p[n]*qxx_m00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n] < 0.0) qx -= .5*qnnn.d_p00*qxx_p00;
        else            qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n] < 0.0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else            qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n] < 0.0) qz -= .5*qnnn.d_00p*qzz_00p;
        else            qz += .5*qnnn.d_00m*qzz_00m;
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

//        if(qx_p00 > 0.0) qx_p00 = 0.0;
//        if(qx_m00 < 0.0) qx_m00 = 0.0;
//        if(qy_0p0 > 0.0) qy_0p0 = 0.0;
//        if(qy_0m0 < 0.0) qy_0m0 = 0.0;
//#ifdef P4_TO_P8
//        if(qz_00p > 0.0) qz_00p = 0.0;
//        if(qz_00m < 0.0) qz_00m = 0.0;
//#endif
//        if(qx_p00 < 0.0) qx_p00 = 0.0;
//        if(qx_m00 > 0.0) qx_m00 = 0.0;
//        if(qy_0p0 < 0.0) qy_0p0 = 0.0;
//        if(qy_0m0 > 0.0) qy_0m0 = 0.0;
//#ifdef P4_TO_P8
//        if(qz_00p < 0.0) qz_00p = 0.0;
//        if(qz_00m > 0.0) qz_00m = 0.0;
//#endif
        double dot_product = SUMD( nx[n]*qx,ny[n]*qy,nz[n]*qz );

        if(cos_angle_p[n] < 0.0) {
          if(qx_p00 > 0.0) qx_p00 = 0.0;
          if(qx_m00 < 0.0) qx_m00 = 0.0;
          if(qy_0p0 > 0.0) qy_0p0 = 0.0;
          if(qy_0m0 < 0.0) qy_0m0 = 0.0;
  #ifdef P4_TO_P8
          if(qz_00p > 0.0) qz_00p = 0.0;
          if(qz_00m < 0.0) qz_00m = 0.0;
  #endif
        } else {
          if(qx_p00 < 0.0) qx_p00 = 0.0;
          if(qx_m00 > 0.0) qx_m00 = 0.0;
          if(qy_0p0 < 0.0) qy_0p0 = 0.0;
          if(qy_0m0 > 0.0) qy_0m0 = 0.0;
  #ifdef P4_TO_P8
          if(qz_00p < 0.0) qz_00p = 0.0;
          if(qz_00m > 0.0) qz_00m = 0.0;
  #endif
        }

        double abs_grad_q = sqrt( SUMD( MAX( qx_p00*qx_p00, qx_m00*qx_m00 ),
                                        MAX( qy_0p0*qy_0p0, qy_0m0*qy_0m0 ),
                                        MAX( qz_00p*qz_00p, qz_00m*qz_00m ) ) );

        tmp_p[n] = q_p[n] - dt*(dot_product - cos_angle_p[n]*abs_grad_q);
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
  if(order != 0 && order != 1 && order != 2) throw std::invalid_argument("[CASL_ERROR]: my_p4est_level_set_t->extend_Over_Interface_TVD: order must be 0, 1 or 2.");
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
    for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      nx[n] = normal_p[0][n];
      ny[n] = normal_p[1][n];
#ifdef P4_TO_P8
      nz[n] = normal_p[2][n];
#endif
    }
  } else {
    for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
    {
      ngbd->get_neighbors(n, qnnn);
      qnnn.gradient(phi_p, DIM(nx[n], ny[n], nz[n]));
      double norm = sqrt(SUMD(nx[n]*nx[n], ny[n]*ny[n], nz[n]*nz[n]));

      if(norm > EPS)
      {
        nx[n] /= norm;
        ny[n] /= norm;
#ifdef P4_TO_P8
        nz[n] /= norm;
#endif
      }
      else
      {
        nx[n] = 0.0;
        ny[n] = 0.0;
#ifdef P4_TO_P8
        nz[n] = 0.0;
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

  if(order >= 1)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &qn); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &b_qn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_p[qnnn.node_000] < -EPS &&
     #ifdef P4_TO_P8
           (phi_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (phi_p[qnnn.node_m00_mp] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           (phi_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (phi_p[qnnn.node_m00_pp] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           (phi_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (phi_p[qnnn.node_p00_mp] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           (phi_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (phi_p[qnnn.node_p00_pp] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           (phi_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (phi_p[qnnn.node_0m0_mp] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           (phi_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (phi_p[qnnn.node_0m0_pp] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           (phi_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (phi_p[qnnn.node_0p0_mp] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           (phi_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (phi_p[qnnn.node_0p0_pp] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           (phi_p[qnnn.node_00m_mm] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (phi_p[qnnn.node_00m_mp] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           (phi_p[qnnn.node_00m_pm] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (phi_p[qnnn.node_00m_pp] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           (phi_p[qnnn.node_00p_mm] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (phi_p[qnnn.node_00p_mp] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           (phi_p[qnnn.node_00p_pm] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (phi_p[qnnn.node_00p_pp] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           (phi_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS) &&
           (phi_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS) &&
           (phi_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS) &&
           (phi_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS) &&
           (phi_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS) &&
           (phi_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS) &&
           (phi_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS) &&
           (phi_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        double grad_q[P4EST_DIM]; qnnn.gradient(q_p, DIM(grad_q[0], grad_q[1], grad_q[2]));
        qn_p[n] = SUMD(nx[n]*grad_q[0], ny[n]*grad_q[1], nz[n]*grad_q[2]);
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0.0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_p[qnnn.node_000] < -EPS &&
     #ifdef P4_TO_P8
           (phi_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (phi_p[qnnn.node_m00_mp] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           (phi_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (phi_p[qnnn.node_m00_pp] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           (phi_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (phi_p[qnnn.node_p00_mp] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           (phi_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (phi_p[qnnn.node_p00_pp] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           (phi_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (phi_p[qnnn.node_0m0_mp] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           (phi_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (phi_p[qnnn.node_0m0_pp] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           (phi_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (phi_p[qnnn.node_0p0_mp] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           (phi_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (phi_p[qnnn.node_0p0_pp] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           (phi_p[qnnn.node_00m_mm] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (phi_p[qnnn.node_00m_mp] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           (phi_p[qnnn.node_00m_pm] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (phi_p[qnnn.node_00m_pp] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           (phi_p[qnnn.node_00p_mm] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (phi_p[qnnn.node_00p_mp] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           (phi_p[qnnn.node_00p_pm] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (phi_p[qnnn.node_00p_pp] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           (phi_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS) &&
           (phi_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS) &&
           (phi_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS) &&
           (phi_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS) &&
           (phi_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS) &&
           (phi_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS) &&
           (phi_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS) &&
           (phi_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        double grad_q[P4EST_DIM]; qnnn.gradient(q_p, DIM(grad_q[0], grad_q[1], grad_q[2]));
        qn_p[n] = SUMD(nx[n]*grad_q[0], ny[n]*grad_q[1], nz[n]*grad_q[2]);
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0.0;
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
    for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000] == true &&
     #ifdef P4_TO_P8
           ( b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_mp] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pp] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mp] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pp] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mp] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pp] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mp] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pp] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00m_mm] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_mp] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pm] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pp] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00p_mm] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_mp] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pm] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pp] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           ( b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        double grad_qn[P4EST_DIM]; qnnn.gradient(qn_p, DIM(grad_qn[0], grad_qn[1], grad_qn[2]));
        qnn_p[n] = SUMD(nx[n]*grad_qn[0], ny[n]*grad_qn[1], nz[n]*grad_qn[2]);
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0.0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000] == true &&
     #ifdef P4_TO_P8
           ( b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_mp] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pp] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mp] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pp] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mp] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pp] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mp] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pp] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00m_mm] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_mp] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pm] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pp] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00p_mm] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_mp] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pm] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pp] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           ( b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        double grad_qn[P4EST_DIM]; qnnn.gradient(qn_p, DIM(grad_qn[0], grad_qn[1], grad_qn[2]));
        qnn_p[n] = SUMD(nx[n]*grad_qn[0], ny[n]*grad_qn[1], nz[n]*grad_qn[2]);
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0.0;
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
  if(order == 2)
  {
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    for(int it = 0; it < iterations; ++it)
    {
      ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnnx = nx[n] > 0.0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                    : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n] > 0.0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                    : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n] > 0.0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                    : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qnn_p[n] - dt*(SUMD(nx[n]*qnnx, ny[n]*qnny, nz[n]*qnnz));
        }
        else
          tmp_p[n] = qnn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnnx = nx[n] > 0.0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                    : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n] > 0.0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                    : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n] > 0.0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                    : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qnn_p[n] - dt*(SUMD(nx[n]*qnnx, ny[n]*qnny, nz[n]*qnnz));
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
  if(order >= 1)
  {
    double *cos_angle_p;

    ierr = VecGetArray(cos_angle, &cos_angle_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);

    foreach_node(n, nodes)
    {
//      b_qn_well_defined_p[n] = true;
      if (phi_p[n] < 0.0)
      {
        qn_p[n] = cos_angle_p[n];
      } else {
        b_qn_well_defined_p[n] = false;
      }
    }

    ierr = VecRestoreArray(cos_angle, &cos_angle_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(qn, &qn_p); CHKERRXX(ierr);

    if(order == 2) { ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr); }
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for(int it = 0; it < iterations; ++it)
    {
      ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnx = nx[n] > 0.0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                                   : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n] > 0.0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                                   : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n] > 0.0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                                   : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qn_p[n] - dt*(SUMD(nx[n]*qnx, ny[n]*qny, nz[n]*qnz)) + (order == 2 ? dt*qnn_p[n] : 0.0);
        }
        else
          tmp_p[n] = qn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnx = nx[n] > 0.0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                                   : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n] > 0.0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                                   : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n] > 0.0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                                   : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qn_p[n] - dt*(SUMD(nx[n]*qnx, ny[n]*qny, nz[n]*qnz)) + (order == 2 ? dt*qnn_p[n] : 0.0);
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

    if(order == 2)
    {
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecDestroy(qnn); CHKERRXX(ierr);
    }
  }

  if(order >= 1) { ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr); }
  if(order == 2) { ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr); }

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

  if(order >= 1) { ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr); }

  for(int it = 0; it < iterations; ++it)
  {
    ngbd->second_derivatives_central(q, DIM(qxx, qyy, qzz));

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_p[n] > -EPS)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
        dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
        dt /= ((double) P4EST_DIM);

        /* first order one sided derivatives */
        double qx = nx[n] > 0.0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n] > 0.0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n] > 0.0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
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
        qxx_m00 = qxx_p[n]*qxx_m00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n] < 0.0) qx -= .5*qnnn.d_p00*qxx_p00;
        else            qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n] < 0.0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else            qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n] < 0.0) qz -= .5*qnnn.d_00p*qzz_00p;
        else            qz += .5*qnnn.d_00m*qzz_00m;
#endif

//#ifdef P4_TO_P8
//        if(fabs(nx[n]) < EPS && fabs(ny[n]) < EPS && fabs(nz[n]) < EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p) +
//                      qnnn.f_00m_linear(q_p) + qnnn.f_00p_linear(q_p))/6.;
//#else
//        if(fabs(nx[n]) < EPS && fabs(ny[n]) < EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p))/4.;
//#endif
//        else
          tmp_p[n] = q_p[n] - dt*(SUMD(nx[n]*qx, ny[n]*qy, nz[n]*qz)) + (order >= 1 ? dt*qn_p[n] : 0.0);
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(phi_p[n] > -EPS)
      {
        double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
        dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
        dt /= ((double) P4EST_DIM);

        /* first order one sided derivatives */
        double qx = nx[n] > 0.0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n] > 0.0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n] > 0.0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
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
        qxx_m00 = qxx_p[n]*qxx_m00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n] < 0.0) qx -= .5*qnnn.d_p00*qxx_p00;
        else            qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n] < 0.0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else            qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n] < 0.0) qz -= .5*qnnn.d_00p*qzz_00p;
        else            qz += .5*qnnn.d_00m*qzz_00m;
#endif

//#ifdef P4_TO_P8
//        if(fabs(nx[n]) < EPS && fabs(ny[n]) < EPS && fabs(nz[n]) < EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p) +
//                      qnnn.f_00m_linear(q_p) + qnnn.f_00p_linear(q_p))/6.;
//#else
//        if(fabs(nx[n]) < EPS && fabs(ny[n]) < EPS)
//          tmp_p[n] = (qnnn.f_m00_linear(q_p) + qnnn.f_p00_linear(q_p) +
//                      qnnn.f_0m0_linear(q_p) + qnnn.f_0p0_linear(q_p))/4.;
//#endif
//        else
          tmp_p[n] = q_p[n] - dt*(SUMD(nx[n]*qx, ny[n]*qy, nz[n]*qz)) + (order >= 1 ? dt*qn_p[n] : 0.0);
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

  if(order >= 1)
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

void my_p4est_level_set_t::advect_in_normal_direction_with_contact_angle(const Vec vn, const Vec surf_tns, const Vec cos_angle, const Vec phi_wall, Vec phi, double dt)
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

  my_p4est_interpolation_nodes_t interp_phi(ngbd);

  Vec phi_new;
  ierr = VecDuplicate(phi, &phi_new); CHKERRXX(ierr);
  VecCopyGhost(phi, phi_new);
  VecAXPBYGhost(phi_new, -dt, 1., vn);
  interp_phi.set_input(phi_new, linear);
  solver.set_wc(dirichlet_cf, interp_phi);
//  solver.set_wc(neumann_cf, zero_cf);

  solver.solve(phi, true);

  ierr = VecDestroy(phi_new); CHKERRXX(ierr);
  ierr = VecDestroy(rhs); CHKERRXX(ierr);
  ierr = VecDestroy(flux); CHKERRXX(ierr);

  double dxyz[P4EST_DIM];
  dxyz_min(p4est, dxyz);

  double dx = MIN(DIM(dxyz[0], dxyz[1], dxyz[2]));

  // extend into wall
  if (phi_wall != NULL && cos_angle != NULL)
  {
    switch (contact_angle_extension)
    {
      case 0: enforce_contact_angle(phi_wall, phi, cos_angle, 50); break;
      case 1: enforce_contact_angle2(phi_wall, phi, cos_angle, 50, 1); break;
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


          double limit_extd = 6.*dx;
          double limit_wall = 6.*dx;

          foreach_node(n, nodes)
          {
            if (fabs(phi_ptr[n]) < limit_extd && phi_wall_ptr[n] < limit_wall)
              region_ptr[n] = 1;
            else
              region_ptr[n] = 0.0;
          }

          ierr = VecRestoreArray(phi, &phi_ptr);           CHKERRXX(ierr);
          ierr = VecRestoreArray(phi_wall, &phi_wall_ptr); CHKERRXX(ierr);
          ierr = VecRestoreArray(region, &region_ptr);     CHKERRXX(ierr);

          extend_Over_Interface_TVD_regional(phi_wall, phi_wall, region, phi, 50, 2); break;
        }
      case 3: extend_Over_Interface_TVD_Full(phi_wall, phi, 50, 2, DBL_MAX, 2000*dx); break;
    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_level_set_advect_in_normal_direction_Vec, 0, 0, 0, 0);
}

void my_p4est_level_set_t::extend_Over_Interface_TVD_regional( Vec phi, Vec mask, Vec region, Vec q, int iterations, int order) const
{
#ifdef CASL_THROWS
  if(order != 0 && order != 1 && order != 2) throw std::invalid_argument("[CASL_ERROR]: my_p4est_level_set_t->extend_Over_Interface_TVD: order must be 0, 1 or 2.");
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
  for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; ++n)
  {
    ngbd->get_neighbors(n, qnnn);
    qnnn.gradient(phi_p, DIM(nx[n], ny[n], nz[n]));
    double norm = sqrt(SUMD(nx[n]*nx[n], ny[n]*ny[n], nz[n]*nz[n]));

    if(norm > EPS)
    {
      nx[n] /= norm;
      ny[n] /= norm;
#ifdef P4_TO_P8
      nz[n] /= norm;
#endif
    }
    else
    {
      nx[n] = 0.0;
      ny[n] = 0.0;
#ifdef P4_TO_P8
      nz[n] = 0.0;
#endif
    }
  }

  ierr = VecGetArray(q , &q_p) ; CHKERRXX(ierr);

  /* initialize qn */
  const std::vector<p4est_locidx_t>& layer_nodes = ngbd->layer_nodes;
  const std::vector<p4est_locidx_t>& local_nodes = ngbd->local_nodes;

  if(order >= 1)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &qn); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &b_qn_well_defined); CHKERRXX(ierr);

    ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr);
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(mask_p[qnnn.node_000] < -EPS &&
     #ifdef P4_TO_P8
           (mask_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (mask_p[qnnn.node_m00_mp] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           (mask_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (mask_p[qnnn.node_m00_pp] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           (mask_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (mask_p[qnnn.node_p00_mp] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           (mask_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (mask_p[qnnn.node_p00_pp] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           (mask_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (mask_p[qnnn.node_0m0_mp] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           (mask_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (mask_p[qnnn.node_0m0_pp] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           (mask_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (mask_p[qnnn.node_0p0_mp] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           (mask_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (mask_p[qnnn.node_0p0_pp] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           (mask_p[qnnn.node_00m_mm] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (mask_p[qnnn.node_00m_mp] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           (mask_p[qnnn.node_00m_pm] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (mask_p[qnnn.node_00m_pp] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           (mask_p[qnnn.node_00p_mm] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (mask_p[qnnn.node_00p_mp] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           (mask_p[qnnn.node_00p_pm] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (mask_p[qnnn.node_00p_pp] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           (mask_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS) &&
           (mask_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS) &&
           (mask_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS) &&
           (mask_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS) &&
           (mask_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS) &&
           (mask_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS) &&
           (mask_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS) &&
           (mask_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        double grad_q[P4EST_DIM]; qnnn.gradient(q_p, DIM(grad_q[0], grad_q[1], grad_q[2]));
        qn_p[n] = SUMD(nx[n]*grad_q[0], ny[n]*grad_q[1], nz[n]*grad_q[2]);
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0.0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(mask_p[qnnn.node_000] < -EPS &&
     #ifdef P4_TO_P8
           (mask_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (mask_p[qnnn.node_m00_mp] < -EPS || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           (mask_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           (mask_p[qnnn.node_m00_pp] < -EPS || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           (mask_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (mask_p[qnnn.node_p00_mp] < -EPS || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           (mask_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           (mask_p[qnnn.node_p00_pp] < -EPS || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           (mask_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (mask_p[qnnn.node_0m0_mp] < -EPS || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           (mask_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           (mask_p[qnnn.node_0m0_pp] < -EPS || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           (mask_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (mask_p[qnnn.node_0p0_mp] < -EPS || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           (mask_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           (mask_p[qnnn.node_0p0_pp] < -EPS || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           (mask_p[qnnn.node_00m_mm] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (mask_p[qnnn.node_00m_mp] < -EPS || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           (mask_p[qnnn.node_00m_pm] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           (mask_p[qnnn.node_00m_pp] < -EPS || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           (mask_p[qnnn.node_00p_mm] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (mask_p[qnnn.node_00p_mp] < -EPS || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           (mask_p[qnnn.node_00p_pm] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           (mask_p[qnnn.node_00p_pp] < -EPS || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           (mask_p[qnnn.node_m00_mm] < -EPS || fabs(qnnn.d_m00_p0) < EPS) &&
           (mask_p[qnnn.node_m00_pm] < -EPS || fabs(qnnn.d_m00_m0) < EPS) &&
           (mask_p[qnnn.node_p00_mm] < -EPS || fabs(qnnn.d_p00_p0) < EPS) &&
           (mask_p[qnnn.node_p00_pm] < -EPS || fabs(qnnn.d_p00_m0) < EPS) &&
           (mask_p[qnnn.node_0m0_mm] < -EPS || fabs(qnnn.d_0m0_p0) < EPS) &&
           (mask_p[qnnn.node_0m0_pm] < -EPS || fabs(qnnn.d_0m0_m0) < EPS) &&
           (mask_p[qnnn.node_0p0_mm] < -EPS || fabs(qnnn.d_0p0_p0) < EPS) &&
           (mask_p[qnnn.node_0p0_pm] < -EPS || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )
      {
        b_qn_well_defined_p[n] = true;
        double grad_q[P4EST_DIM]; qnnn.gradient(q_p, DIM(grad_q[0], grad_q[1], grad_q[2]));
        qn_p[n] = SUMD(nx[n]*grad_q[0], ny[n]*grad_q[1], nz[n]*grad_q[2]);
      }
      else
      {
        b_qn_well_defined_p[n] = false;
        qn_p[n] = 0.0;
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
    for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000] == true &&
     #ifdef P4_TO_P8
           ( b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_mp] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pp] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mp] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pp] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mp] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pp] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mp] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pp] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00m_mm] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_mp] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pm] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pp] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00p_mm] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_mp] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pm] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pp] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           ( b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        double grad_qn[P4EST_DIM]; qnnn.gradient(qn_p, DIM(grad_qn[0], grad_qn[1], grad_qn[2]));
        qnn_p[n] = SUMD(nx[n]*grad_qn[0], ny[n]*grad_qn[1], nz[n]*grad_qn[2]);
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0.0;
      }
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(b_qnn_well_defined, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(qnn, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(  b_qn_well_defined_p[qnnn.node_000] == true &&
     #ifdef P4_TO_P8
           ( b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_mp] == true || fabs(qnnn.d_m00_p0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pp] == true || fabs(qnnn.d_m00_m0) < EPS || fabs(qnnn.d_m00_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mp] == true || fabs(qnnn.d_p00_p0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pp] == true || fabs(qnnn.d_p00_m0) < EPS || fabs(qnnn.d_p00_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mp] == true || fabs(qnnn.d_0m0_p0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pp] == true || fabs(qnnn.d_0m0_m0) < EPS || fabs(qnnn.d_0m0_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mp] == true || fabs(qnnn.d_0p0_p0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pp] == true || fabs(qnnn.d_0p0_m0) < EPS || fabs(qnnn.d_0p0_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00m_mm] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_mp] == true || fabs(qnnn.d_00m_p0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pm] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00m_pp] == true || fabs(qnnn.d_00m_m0) < EPS || fabs(qnnn.d_00m_0m) < EPS) &&

           ( b_qn_well_defined_p[qnnn.node_00p_mm] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_mp] == true || fabs(qnnn.d_00p_p0) < EPS || fabs(qnnn.d_00p_0m) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pm] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0p) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_00p_pp] == true || fabs(qnnn.d_00p_m0) < EPS || fabs(qnnn.d_00p_0m) < EPS)
     #else
           ( b_qn_well_defined_p[qnnn.node_m00_mm] == true || fabs(qnnn.d_m00_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_m00_pm] == true || fabs(qnnn.d_m00_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_mm] == true || fabs(qnnn.d_p00_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_p00_pm] == true || fabs(qnnn.d_p00_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_mm] == true || fabs(qnnn.d_0m0_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0m0_pm] == true || fabs(qnnn.d_0m0_m0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_mm] == true || fabs(qnnn.d_0p0_p0) < EPS) &&
           ( b_qn_well_defined_p[qnnn.node_0p0_pm] == true || fabs(qnnn.d_0p0_m0) < EPS)
     #endif
           )


      {
        b_qnn_well_defined_p[n] = true;
        double grad_qn[P4EST_DIM]; qnnn.gradient(qn_p, DIM(grad_qn[0], grad_qn[1], grad_qn[2]));
        qnn_p[n] = SUMD(nx[n]*grad_qn[0], ny[n]*grad_qn[1], nz[n]*grad_qn[2]);
      }
      else
      {
        b_qnn_well_defined_p[n] = false;
        qnn_p[n] = 0.0;
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
  if(order == 2)
  {
    ierr = VecGetArray(b_qnn_well_defined, &b_qnn_well_defined_p); CHKERRXX(ierr);

    for(int it = 0; it < iterations; ++it)
    {
      ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnnx = nx[n] > 0.0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                    : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n] > 0.0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                    : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n] > 0.0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                    : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qnn_p[n] - dt*(SUMD(nx[n]*qnnx, ny[n]*qnny, nz[n]*qnnz));
        }
        else
          tmp_p[n] = qnn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qnn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnnx = nx[n] > 0.0 ? (qnn_p[n] - qnnn.f_m00_linear(qnn_p)) / qnnn.d_m00
                                    : (qnnn.f_p00_linear(qnn_p) - qnn_p[n]) / qnnn.d_p00;
          double qnny = ny[n] > 0.0 ? (qnn_p[n] - qnnn.f_0m0_linear(qnn_p)) / qnnn.d_0m0
                                    : (qnnn.f_0p0_linear(qnn_p) - qnn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnnz = nz[n] > 0.0 ? (qnn_p[n] - qnnn.f_00m_linear(qnn_p)) / qnnn.d_00m
                                    : (qnnn.f_00p_linear(qnn_p) - qnn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qnn_p[n] - dt*(SUMD(nx[n]*qnnx, ny[n]*qnny, nz[n]*qnnz));
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
  if(order >= 1)
  {
    if(order == 2) { ierr = VecGetArray(qnn, &qnn_p); CHKERRXX(ierr); }
    ierr = VecGetArray(b_qn_well_defined, &b_qn_well_defined_p); CHKERRXX(ierr);

    for(int it = 0; it < iterations; ++it)
    {
      ierr = VecGetArray(qn , &qn_p ); CHKERRXX(ierr);
      ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

      /* first do the layer nodes */
      for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = layer_nodes[n_map];
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnx = nx[n] > 0.0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                                   : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n] > 0.0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                                   : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n] > 0.0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                                   : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qn_p[n] - dt*(SUMD(nx[n]*qnx, ny[n]*qny, nz[n]*qnz)) + (order == 2 ? dt*qnn_p[n]*region_p[n] : 0.0);
        }
        else
          tmp_p[n] = qn_p[n];
      }

      /* initiate the communication */
      ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      /* now process the local nodes */
      for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
      {
        p4est_locidx_t n = local_nodes[n_map];
        if(!b_qn_well_defined_p[n])
        {
          ngbd->get_neighbors(n, qnnn);
          double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
          dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
          dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
          dt /= ((double) P4EST_DIM);

          /* first order one sided derivative */
          double qnx = nx[n] > 0.0 ? (qn_p[n] - qnnn.f_m00_linear(qn_p)) / qnnn.d_m00
                                   : (qnnn.f_p00_linear(qn_p) - qn_p[n]) / qnnn.d_p00;
          double qny = ny[n] > 0.0 ? (qn_p[n] - qnnn.f_0m0_linear(qn_p)) / qnnn.d_0m0
                                   : (qnnn.f_0p0_linear(qn_p) - qn_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
          double qnz = nz[n] > 0.0 ? (qn_p[n] - qnnn.f_00m_linear(qn_p)) / qnnn.d_00m
                                   : (qnnn.f_00p_linear(qn_p) - qn_p[n]) / qnnn.d_00p;
#endif
          tmp_p[n] = qn_p[n] - dt*(SUMD(nx[n]*qnx, ny[n]*qny, nz[n]*qnz)) + (order == 2 ? dt*qnn_p[n]*region_p[n] : 0.0);
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

    if(order == 2)
    {
      ierr = VecRestoreArray(qnn, &qnn_p); CHKERRXX(ierr);
      ierr = VecDestroy(qnn); CHKERRXX(ierr);
    }
  }

  if(order >= 1) { ierr = VecDestroy(b_qn_well_defined ); CHKERRXX(ierr); }
  if(order == 2) { ierr = VecDestroy(b_qnn_well_defined); CHKERRXX(ierr); }

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

  if(order >= 1) { ierr = VecGetArray(qn, &qn_p); CHKERRXX(ierr); }

  for(int it = 0; it < iterations; ++it)
  {
    ngbd->second_derivatives_central(q, DIM(qxx, qyy, qzz));

    ierr = VecGetArray(qxx, &qxx_p); CHKERRXX(ierr);
    ierr = VecGetArray(qyy, &qyy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(qzz, &qzz_p); CHKERRXX(ierr);
#endif

    ierr = VecGetArray(q  , &q_p  ); CHKERRXX(ierr);
    ierr = VecGetArray(tmp, &tmp_p); CHKERRXX(ierr);

    /* first do the layer nodes */
    for(size_t n_map = 0; n_map < layer_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = layer_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(mask_p[n] > -EPS)
      {
        const quad_neighbor_nodes_of_node_t& qnnn = (*ngbd)[n];
        double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
        dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
        dt /= ((double) P4EST_DIM);

        /* first order one sided derivatives */
        double qx = nx[n] > 0.0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n] > 0.0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n] > 0.0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
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
        qxx_m00 = qxx_p[n]*qxx_m00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n] < 0.0) qx -= .5*qnnn.d_p00*qxx_p00;
        else            qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n] < 0.0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else            qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n] < 0.0) qz -= .5*qnnn.d_00p*qzz_00p;
        else            qz += .5*qnnn.d_00m*qzz_00m;
#endif

        tmp_p[n] = q_p[n] - dt*(SUMD(nx[n]*qx, ny[n]*qy, nz[n]*qz)) + (order >= 1 ? dt*qn_p[n]*region_p[n] : 0.0);
      }
      else
        tmp_p[n] = q_p[n];
    }

    /* initiate the communication */
    ierr = VecGhostUpdateBegin(tmp, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now process the local nodes */
    for(size_t n_map = 0; n_map < local_nodes.size(); ++n_map)
    {
      p4est_locidx_t n = local_nodes[n_map];
      ngbd->get_neighbors(n, qnnn);
      if(mask_p[n] > -EPS)
      {
        double dt = MIN(fabs(qnnn.d_m00), fabs(qnnn.d_p00) );
        dt  =  MIN(dt, fabs(qnnn.d_0m0), fabs(qnnn.d_0p0) );
#ifdef P4_TO_P8
        dt  =  MIN(dt, fabs(qnnn.d_00m), fabs(qnnn.d_00p) );
#endif
        dt /= ((double) P4EST_DIM);

        /* first order one sided derivatives */
        double qx = nx[n] > 0.0 ? (q_p[n] - qnnn.f_m00_linear(q_p)) / qnnn.d_m00
                                : (qnnn.f_p00_linear(q_p) - q_p[n]) / qnnn.d_p00;
        double qy = ny[n] > 0.0 ? (q_p[n] - qnnn.f_0m0_linear(q_p)) / qnnn.d_0m0
                                : (qnnn.f_0p0_linear(q_p) - q_p[n]) / qnnn.d_0p0;
#ifdef P4_TO_P8
        double qz = nz[n] > 0.0 ? (q_p[n] - qnnn.f_00m_linear(q_p)) / qnnn.d_00m
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
        qxx_m00 = qxx_p[n]*qxx_m00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_m00) ? qxx_p[n] : qxx_m00);
        qxx_p00 = qxx_p[n]*qxx_p00 < 0.0 ? 0.0 : (fabs(qxx_p[n]) < fabs(qxx_p00) ? qxx_p[n] : qxx_p00);
        qyy_0m0 = qyy_p[n]*qyy_0m0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0m0) ? qyy_p[n] : qyy_0m0);
        qyy_0p0 = qyy_p[n]*qyy_0p0 < 0.0 ? 0.0 : (fabs(qyy_p[n]) < fabs(qyy_0p0) ? qyy_p[n] : qyy_0p0);
#ifdef P4_TO_P8
        qzz_00m = qzz_p[n]*qzz_00m < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00m) ? qzz_p[n] : qzz_00m);
        qzz_00p = qzz_p[n]*qzz_00p < 0.0 ? 0.0 : (fabs(qzz_p[n]) < fabs(qzz_00p) ? qzz_p[n] : qzz_00p);
#endif

        if(nx[n] < 0.0) qx -= .5*qnnn.d_p00*qxx_p00;
        else            qx += .5*qnnn.d_m00*qxx_m00;
        if(ny[n] < 0.0) qy -= .5*qnnn.d_0p0*qyy_0p0;
        else            qy += .5*qnnn.d_0m0*qyy_0m0;
#ifdef P4_TO_P8
        if(nz[n] < 0.0) qz -= .5*qnnn.d_00p*qzz_00p;
        else            qz += .5*qnnn.d_00m*qzz_00m;
#endif

        tmp_p[n] = q_p[n] - dt*(SUMD(nx[n]*qx, ny[n]*qy, nz[n]*qz)) + (order >= 1 ? dt*qn_p[n]*region_p[n] : 0.0);
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

  if(order >= 1)
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

