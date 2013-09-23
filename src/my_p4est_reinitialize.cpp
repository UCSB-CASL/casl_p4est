#include "my_p4est_reinitialize.h"
#include "petsc_compatibility.h"
#include "point2.h"
#include "interpolating_function.h"
#include "refine_coarsen.h"


void my_p4est_level_set::reinitialize_One_Iteration_First_Order( std::vector<p4est_locidx_t>& map, double *p0, double *pn, double *pnp1, double limit )
{
  for( size_t n_map=0; n_map<map.size(); ++n_map)
  {
    p4est_locidx_t n = map[n_map];

    if(fabs(p0[n]) <= EPS)
      pnp1[n] = 0;
    else if(fabs(p0[n]) <= limit)
    {
      double p0_00, p0_m0, p0_p0, p0_0m, p0_0p ;
      double p_00 , p_m0 , p_p0 , p_0m , p_0p  ;
      (*ngbd)[n].ngbd_with_quadratic_interpolation(p0, p0_00, p0_m0, p0_p0, p0_0m, p0_0p);
      (*ngbd)[n].ngbd_with_quadratic_interpolation(pn, p_00 , p_m0 , p_p0 , p_0m , p_0p );
      double s_p0 = (*ngbd)[n].d_p0; double s_m0 = (*ngbd)[n].d_m0;
      double s_0p = (*ngbd)[n].d_0p; double s_0m = (*ngbd)[n].d_0m;

      //---------------------------------------------------------------------
      // check if the node is near interface
      //---------------------------------------------------------------------
      if ( (p0_00*p0_m0<0) || (p0_00*p0_p0<0) || (p0_00*p0_0m<0) || (p0_00*p0_0p<0) )
      {
        if(p0_00*p0_m0<0) { s_m0 = -interface_Location(-s_m0, 0, p0_m0, p0_00); p_m0 = 0; }
        if(p0_00*p0_p0<0) { s_p0 =  interface_Location( s_p0, 0, p0_p0, p0_00); p_p0 = 0; }
        if(p0_00*p0_0m<0) { s_0m = -interface_Location(-s_0m, 0, p0_0m, p0_00); p_0m = 0; }
        if(p0_00*p0_0p<0) { s_0p =  interface_Location( s_0p, 0, p0_0p, p0_00); p_0p = 0; }

        s_m0 = MAX(s_m0,EPS);
        s_p0 = MAX(s_p0,EPS);
        s_0m = MAX(s_0m,EPS);
        s_0p = MAX(s_0p,EPS);
      }

      //---------------------------------------------------------------------
      // Neumann boundary condition on the walls
      //---------------------------------------------------------------------
      /* first unclamp the node */
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n+nodes->offset_owned_indeps);
      p4est_indep_t unclamped_node = *node;
      p4est_node_unclamp((p4est_quadrant_t*)&unclamped_node);

      /* wall in the x direction */
      if(unclamped_node.x==0)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
        if(nb_tree_idx == tree_idx) { s_m0 = s_p0; p_m0 = p_p0; }
      }
      else if(unclamped_node.x==P4EST_ROOT_LEN)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
        if(nb_tree_idx == tree_idx) { s_p0 = s_m0; p_p0 = p_m0; }
      }

      /* wall in the y direction */
      if(unclamped_node.y==0)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 2];
        if(nb_tree_idx == tree_idx) { s_0m = s_0p; p_0m = p_0p; }
      }
      else if(unclamped_node.y==P4EST_ROOT_LEN)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 3];
        if(nb_tree_idx == tree_idx) { s_0p = s_0m; p_0p = p_0m; }
      }

      //---------------------------------------------------------------------
      // First Order One-Sided Differecing
      //---------------------------------------------------------------------
      double px_p0 = (p_p0-p_00)/s_p0; double px_m0 = (p_00-p_m0)/s_m0;
      double py_0p = (p_0p-p_00)/s_0p; double py_0m = (p_00-p_0m)/s_0m;

      double sgn = (p0_00>0) ? 1 : -1;

      //---------------------------------------------------------------------
      // Upwind Scheme
      //---------------------------------------------------------------------
      double dt = MIN(s_m0,s_p0);
      dt = MIN(dt,s_0m);
      dt = MIN(dt,s_0p);
      dt = dt/2.;

      if(sgn>0) { if(px_p0>0) px_p0 = 0; if(px_m0<0) px_m0 = 0; if(py_0p>0) py_0p = 0; if(py_0m<0) py_0m = 0;}
      else      { if(px_p0<0) px_p0 = 0; if(px_m0>0) px_m0 = 0; if(py_0p<0) py_0p = 0; if(py_0m>0) py_0m = 0;}

      pnp1[n] = p_00 - dt*sgn*(sqrt( MAX(px_p0*px_p0 , px_m0*px_m0) + MAX( py_0p*py_0p , py_0m*py_0m ) ) - 1.);
      if(p0_00*pnp1[n]<0) pnp1[n] *= -1;
    }
    /* else : far away from the interface and not in the band ... nothing to do */
    else
      pnp1[n] = p0[n];
  }
}


void my_p4est_level_set::reinitialize_One_Iteration_Second_Order( std::vector<p4est_locidx_t>& map, const double *dxx0, const double *dyy0, const double *dxx, const double *dyy, double *p0, double *pn, double *pnp1, double limit )
{
  for( size_t n_map=0; n_map<map.size(); ++n_map)
  {
    p4est_locidx_t n = map[n_map];

    if(n!=n) printf("asadfsdf\n");
    if(fabs(p0[n]) <= EPS)
      pnp1[n] = 0;
    else if(fabs(p0[n]) <= limit)
    {
      double p0_00, p0_m0, p0_p0, p0_0m, p0_0p ;
      double p_00 , p_m0 , p_p0 , p_0m , p_0p  ;
      (*ngbd)[n].ngbd_with_quadratic_interpolation(p0, p0_00, p0_m0, p0_p0, p0_0m, p0_0p);
      (*ngbd)[n].ngbd_with_quadratic_interpolation(pn, p_00 , p_m0 , p_p0 , p_0m , p_0p );
      double s_p0 = (*ngbd)[n].d_p0; double s_m0 = (*ngbd)[n].d_m0;
      double s_0p = (*ngbd)[n].d_0p; double s_0m = (*ngbd)[n].d_0m;

      //---------------------------------------------------------------------
      // Second Order derivatives
      //---------------------------------------------------------------------
      double pxx_00 = dxx[n];
      double pyy_00 = dyy[n];
      double pxx_m0 = (*ngbd)[n].f_m0_linear(dxx);
      double pxx_p0 = (*ngbd)[n].f_p0_linear(dxx);
      double pyy_0m = (*ngbd)[n].f_0m_linear(dyy);
      double pyy_0p = (*ngbd)[n].f_0p_linear(dyy);

      //---------------------------------------------------------------------
      // check if the node is near interface
      //---------------------------------------------------------------------
      if ( (p0_00*p0_m0<0) || (p0_00*p0_p0<0) || (p0_00*p0_0m<0) || (p0_00*p0_0p<0) )
      {
        double p0xx_00 = dxx0[n];
        double p0yy_00 = dyy0[n];
        double p0xx_m0 = (*ngbd)[n].f_m0_linear(dxx0);
        double p0xx_p0 = (*ngbd)[n].f_p0_linear(dxx0);
        double p0yy_0m = (*ngbd)[n].f_0m_linear(dyy0);
        double p0yy_0p = (*ngbd)[n].f_0p_linear(dyy0);

        if(p0_00*p0_m0<0) { s_m0 =-interface_Location_With_Second_Order_Derivative(-s_m0,   0,p0_m0,p0_00,p0xx_m0,p0xx_00); p_m0=0; }
        if(p0_00*p0_p0<0) { s_p0 = interface_Location_With_Second_Order_Derivative(    0,s_p0,p0_00,p0_p0,p0xx_00,p0xx_p0); p_p0=0; }
        if(p0_00*p0_0m<0) { s_0m =-interface_Location_With_Second_Order_Derivative(-s_0m,   0,p0_0m,p0_00,p0yy_0m,p0yy_00); p_0m=0; }
        if(p0_00*p0_0p<0) { s_0p = interface_Location_With_Second_Order_Derivative(    0,s_0p,p0_00,p0_0p,p0yy_00,p0yy_0p); p_0p=0; }

        //        if(p0_00*p0_m0<0) { s_m0 = -interface_Location_With_First_Order_Derivative(-s_m0, 0   , p0_m0, p0_00, p0x_m0, p0x_00); p_m0 = 0; }
        //        if(p0_00*p0_p0<0) { s_p0 =  interface_Location_With_First_Order_Derivative(    0, s_p0, p0_00, p0_p0, p0x_00, p0x_p0); p_p0 = 0; }
        //        if(p0_00*p0_0m<0) { s_0m = -interface_Location_With_First_Order_Derivative(-s_0m, 0   , p0_0m, p0_00, p0y_0m, p0y_00); p_0m = 0; }
        //        if(p0_00*p0_0p<0) { s_0p =  interface_Location_With_First_Order_Derivative(    0, s_0p, p0_00, p0_0p, p0y_00, p0y_0p); p_0p = 0; }

        s_m0 = MAX(s_m0,EPS);
        s_p0 = MAX(s_p0,EPS);
        s_0m = MAX(s_0m,EPS);
        s_0p = MAX(s_0p,EPS);
      }

      //---------------------------------------------------------------------
      // Neumann boundary condition on the walls
      //---------------------------------------------------------------------
      /* first unclamp the node */
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes,n+nodes->offset_owned_indeps);
      p4est_indep_t unclamped_node = *node;
      p4est_node_unclamp((p4est_quadrant_t*)&unclamped_node);

      /* wall in the x direction */
      if(unclamped_node.x==0)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 0];
        if(nb_tree_idx == tree_idx) { s_m0 = s_p0; p_m0=p_p0; pxx_00 = pxx_m0 = pxx_p0 = 0; }
      }
      else if(unclamped_node.x==P4EST_ROOT_LEN)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 1];
        if(nb_tree_idx == tree_idx) { s_p0 = s_m0; p_p0=p_m0; pxx_00 = pxx_m0 = pxx_p0 = 0; }
      }

      /* wall in the y direction */
      if(unclamped_node.y==0)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 2];
        if(nb_tree_idx == tree_idx) { s_0m = s_0p; p_0m=p_0p; pyy_00 = pyy_0m = pyy_0p = 0; }
      }
      else if(unclamped_node.y==P4EST_ROOT_LEN)
      {
        p4est_topidx_t tree_idx = node->p.piggy3.which_tree;
        p4est_topidx_t nb_tree_idx = p4est->connectivity->tree_to_tree[2*P4EST_DIM*tree_idx + 3];
        if(nb_tree_idx == tree_idx) { s_0p = s_0m; p_0p=p_0m; pyy_00 = pyy_0m = pyy_0p = 0; }
      }

      //---------------------------------------------------------------------
      // First Order One-Sided Differecing
      //---------------------------------------------------------------------
      double px_p0 = (p_p0-p_00)/s_p0; double px_m0 = (p_00-p_m0)/s_m0;
      double py_0p = (p_0p-p_00)/s_0p; double py_0m = (p_00-p_0m)/s_0m;

      //---------------------------------------------------------------------
      // Second Order One-Sided Differencing
      //---------------------------------------------------------------------
      pxx_m0 = MINMOD(pxx_m0,pxx_00);   px_m0 += 0.5*s_m0*(pxx_m0);
      pxx_p0 = MINMOD(pxx_p0,pxx_00);   px_p0 -= 0.5*s_p0*(pxx_p0);
      pyy_0m = MINMOD(pyy_0m,pyy_00);   py_0m += 0.5*s_0m*(pyy_0m);
      pyy_0p = MINMOD(pyy_0p,pyy_00);   py_0p -= 0.5*s_0p*(pyy_0p);

      double sgn = (p0_00>0) ? 1 : -1;

      //---------------------------------------------------------------------
      // Upwind Scheme
      //---------------------------------------------------------------------
      double dt = MIN(s_m0,s_p0);
      dt = MIN(dt,s_0m);
      dt = MIN(dt,s_0p);
      dt = dt/2.;

      if(sgn>0) { if(px_p0>0) px_p0 = 0; if(px_m0<0) px_m0 = 0; if(py_0p>0) py_0p = 0; if(py_0m<0) py_0m = 0;}
      else      { if(px_p0<0) px_p0 = 0; if(px_m0>0) px_m0 = 0; if(py_0p<0) py_0p = 0; if(py_0m>0) py_0m = 0;}

      pnp1[n] = p_00 - dt*sgn*(sqrt( MAX(px_p0*px_p0 , px_m0*px_m0) + MAX( py_0p*py_0p , py_0m*py_0m ) ) - 1.);
      if(p0_00*pnp1[n]<0) pnp1[n] *= -1;
    }
    /* else : far away from the interface and not in the band ... nothing to do */
    else
      pnp1[n] = p0[n];
  }
}



void my_p4est_level_set::reinitialize_1st_order( Vec &phi_petsc, int number_of_iteration, double limit )
{
  PetscErrorCode ierr;
  double *phi;

  double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
  double *p1 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));

  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    p0[n] = phi[n];
  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

  for(int i=0; i<number_of_iteration; i++)
  {
    /* process the local nodes */
    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    reinitialize_One_Iteration_First_Order( local_nodes, p0, phi, p1, limit);
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

    /* finish receiving the ghost layer */
    if(i!=0) { ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

    /* processes the layer nodes */
    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    reinitialize_One_Iteration_First_Order( layer_nodes, p0, phi, p1, limit);
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      phi[n] = p1[n];
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);


    /* initiate the communication for the ghost layer */
    ierr = VecGhostUpdateBegin(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  /* synchronize */
  ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  free(p0);
  free(p1);
}






void my_p4est_level_set::reinitialize_2nd_order_time_1st_order_space( Vec &phi_petsc, int number_of_iteration, double limit )
{
  /* let's call
     * Ln the local nodes at time n
     * Bnp1 the boundary nodes at time np1
     * Gnm1 the ghost nodes at time nm1
     */
  PetscErrorCode ierr;
  double *phi;

  double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
  double *p1 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
  double *p2 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));

  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    p0[n] = phi[n];
  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

  for(int i=0; i<number_of_iteration; i++)
  {
    /* p1(Ln, Gn, Gnm1) */
    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      p1[n] = phi[n];

    /* process the local nodes */
    /* phi(Lnp1, Bn, Gnm1) */
    reinitialize_One_Iteration_First_Order( local_nodes, p0, p1, phi, limit);
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

    /* finish receiving the ghost layer */
    /* phi(Lnp1, Bn, Gn) */
    if(i!=0) { ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

    /* processes the layer nodes */
    /* p1(Ln, Bn, Gn) */
    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    for(size_t n=nodes->num_owned_indeps; n<nodes->indep_nodes.elem_count; ++n)
      p1[n] = phi[n];
    /* phi(Lnp1, Bnp1, Gn) */
    reinitialize_One_Iteration_First_Order( layer_nodes, p0, p1, phi, limit);
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

    /* initiate the communication for the ghost layer */
    ierr = VecGhostUpdateBegin(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);


    /* process the local nodes */
    /* p2(Lnp2, Bnp1, Gn) */
    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    reinitialize_One_Iteration_First_Order( local_nodes, p0, phi, p2, limit);
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

    /* finish receiving the ghost layer */
    /* phi(Lnp1, Bnp1, Gnp1) */
    ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* process the layer nodes */
    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    /* p2(Lnp2, Bnp2, Gn) */
    reinitialize_One_Iteration_First_Order( layer_nodes, p0, phi, p2, limit);

    /* update phi */
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      phi[n] = .5 * (p1[n] + p2[n]);
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

    /* initiate the communication for the ghost layer */
    ierr = VecGhostUpdateBegin(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  /* synchronize */
  if(number_of_iteration>0) { ierr = VecGhostUpdateEnd(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr); }

  free(p0);
  free(p1);
  free(p2);
}



void my_p4est_level_set::compute_derivatives( Vec &phi_petsc, Vec &dxx_petsc, Vec &dyy_petsc) const
{
  /* first compute dx and dy */
  double *dxx, *dyy;
  PetscErrorCode ierr;
  ierr = VecGetArray(dxx_petsc, &dxx); CHKERRXX(ierr);
  ierr = VecGetArray(dyy_petsc, &dyy); CHKERRXX(ierr);

  double *phi;
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    dxx[n] = (*ngbd)[n].dxx_central(phi);

  ierr = VecGhostUpdateBegin(dxx_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    dyy[n] = (*ngbd)[n].dyy_central(phi);

  ierr = VecGhostUpdateEnd  (dxx_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(dyy_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (dyy_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

  ierr = VecRestoreArray(dxx_petsc, &dxx); CHKERRXX(ierr);
  ierr = VecRestoreArray(dyy_petsc, &dyy); CHKERRXX(ierr);

}



void my_p4est_level_set::reinitialize_2nd_order( Vec &phi_petsc, int number_of_iteration, double limit )
{
  /* let's call
     * Ln the local nodes at time n
     * Bnp1 the boundary nodes at time np1
     * Gnm1 the ghost nodes at time nm1
     */
  PetscErrorCode ierr;
  double *phi;

  double *p0 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
  double *p1 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));
  double *p2 = (double*) malloc(nodes->indep_nodes.elem_count * sizeof(double));

  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
    p0[n] = phi[n];
  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

  Vec dxx0_petsc, dyy0_petsc;
  double *dxx0, *dyy0;
  ierr = VecDuplicate( phi_petsc, &dxx0_petsc); CHKERRXX(ierr);
  ierr = VecDuplicate( phi_petsc, &dyy0_petsc); CHKERRXX(ierr);

  Vec dxx_petsc, dyy_petsc;
  double *dxx, *dyy;
  ierr = VecDuplicate( phi_petsc, &dxx_petsc); CHKERRXX(ierr);
  ierr = VecDuplicate( phi_petsc, &dyy_petsc); CHKERRXX(ierr);

  compute_derivatives(phi_petsc, dxx0_petsc, dyy0_petsc);

  ierr = VecGetArray(dxx0_petsc, &dxx0); CHKERRXX(ierr);
  ierr = VecGetArray(dyy0_petsc, &dyy0); CHKERRXX(ierr);

  for(int i=0; i<number_of_iteration; i++)
  {
    /* p1(Ln, Gn, Gn) */
    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
      p1[n] = phi[n];

    compute_derivatives(phi_petsc, dxx_petsc, dyy_petsc);
    ierr = VecGetArray(dxx_petsc, &dxx); CHKERRXX(ierr);
    ierr = VecGetArray(dyy_petsc, &dyy); CHKERRXX(ierr);

    reinitialize_One_Iteration_Second_Order( local_nodes, dxx0, dyy0, dxx, dyy, p0, p1, phi, limit);
    reinitialize_One_Iteration_Second_Order( layer_nodes, dxx0, dyy0, dxx, dyy, p0, p1, phi, limit);
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

    /* initiate the communication for the ghost layer */
    ierr = VecGhostUpdateBegin(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* now phi(Lnp1, Bnp1, Gnp1) */
    compute_derivatives(phi_petsc, dxx_petsc, dyy_petsc);
    ierr = VecGetArray(dxx_petsc, &dxx); CHKERRXX(ierr);
    ierr = VecGetArray(dyy_petsc, &dyy); CHKERRXX(ierr);

    ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);
    reinitialize_One_Iteration_Second_Order( local_nodes, dxx0, dyy0, dxx, dyy, p0, phi, p2, limit);
    reinitialize_One_Iteration_Second_Order( layer_nodes, dxx0, dyy0, dxx, dyy, p0, phi, p2, limit);

    /* update phi */
    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
      phi[n] = .5 * (p1[n] + p2[n]);
    ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);

    /* initiate the communication for the ghost layer */
    ierr = VecGhostUpdateBegin(phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (phi_petsc, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  ierr = VecRestoreArray(dxx0_petsc, &dxx0); CHKERRXX(ierr);
  ierr = VecRestoreArray(dyy0_petsc, &dyy0); CHKERRXX(ierr);
  ierr = VecDestroy(dxx0_petsc); CHKERRXX(ierr);
  ierr = VecDestroy(dyy0_petsc); CHKERRXX(ierr);

  free(p0);
  free(p1);
  free(p2);
}

void my_p4est_level_set::extend_Over_Interface( Vec &phi_petsc, Vec &q_petsc, BoundaryConditions2D &bc, int order, int band_to_extend ) const
{
#ifdef CASL_THROWS
  if(bc.interfaceType()==NOINTERFACE) throw std::invalid_argument("[CASL_ERROR]: extend_over_interface: no interface defined in the boundary condition ... needs to be dirichlet or neumann.");
  if(order!=0 && order!=1 && order!=2) throw std::invalid_argument("[CASL_ERROR]: extend_over_interface: invalid order. Choose 0, 1 or 2");
#endif
  PetscErrorCode ierr;

  double *phi;
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

  /* first compute the phi derivatives */
  std::vector<double> phi_x(nodes->num_owned_indeps);
  std::vector<double> phi_y(nodes->num_owned_indeps);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    phi_x[n] = (*ngbd)[n].dx_central(phi);
    phi_y[n] = (*ngbd)[n].dy_central(phi);
  }

  InterpolatingFunction interp1(p4est, nodes, ghost, myb, ngbd);
  InterpolatingFunction interp2(p4est, nodes, ghost, myb, ngbd);

  /* find dx and dy smallest */
  p4est_topidx_t tm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t tp = p4est->connectivity->tree_to_vertex[0 + 3];

  double xmin = p4est->connectivity->vertices[3*tm + 0];
  double ymin = p4est->connectivity->vertices[3*tm + 1];
  double xmax = p4est->connectivity->vertices[3*tp + 0];
  double ymax = p4est->connectivity->vertices[3*tp + 1];


  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  double dx = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  double dy = (ymax-ymin) / pow(2.,(double) data->max_lvl);
  double diag = sqrt(dx*dx + dy*dy);

  std::vector<double> q0;
  std::vector<double> q1;
  std::vector<double> q2;

  if(bc.interfaceType()==DIRICHLET)                           q0.resize(nodes->num_owned_indeps);
  if(order >= 1 || (order==0 && bc.interfaceType()==NEUMANN)) q1.resize(nodes->num_owned_indeps);
  if(order >= 2)                                              q2.resize(nodes->num_owned_indeps);

  /* now buffer the interpolation points */
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    Point2 grad_phi;
    grad_phi.x = -phi_x[n];
    grad_phi.y = -phi_y[n];

    if(phi[n]>0 && phi[n]<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      grad_phi /= grad_phi.norm_L2();
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];

      Point2 p_C;
      p_C.x = int2double_coordinate_transform(node->x) + tree_xmin;
      p_C.y = int2double_coordinate_transform(node->y) + tree_ymin;

      if(bc.interfaceType()==DIRICHLET)
        q0[n] = bc.interfaceValue(p_C.x + grad_phi.x*phi[n], p_C.y + grad_phi.y*phi[n]);

      if(order >= 1 || (order==0 && bc.interfaceType()==NEUMANN))
      {
        double x = p_C.x + grad_phi.x * (diag + phi[n]);
        double y = p_C.y + grad_phi.y * (diag + phi[n]);
        interp1.add_point_to_buffer(n, x, y);
      }

      if(order >= 2)
      {
        double x = p_C.x + grad_phi.x * (2*diag + phi[n]);
        double y = p_C.y + grad_phi.y * (2*diag + phi[n]);
        interp2.add_point_to_buffer(n, x, y);
      }
    }
  }

  interp1.set_input_parameters(q_petsc, quadratic);
  interp2.set_input_parameters(q_petsc, quadratic);

  interp1.interpolate(q1.data());
  interp2.interpolate(q2.data());

  /* now compute the extrapolated values */
  double *q;
  ierr = VecGetArray(q_petsc, &q); CHKERRXX(ierr);
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    Point2 grad_phi;
    grad_phi.x = phi_x[n];
    grad_phi.y = phi_y[n];

    if(phi[n]>0 && phi[n]<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      grad_phi /= grad_phi.norm_L2();
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];

      double x = int2double_coordinate_transform(node->x) + tree_xmin;
      double y = int2double_coordinate_transform(node->y) + tree_ymin;

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
          double dif01 = (q1[n] - q0[n])/(diag - 0);
          q[n] = q0[n] + (-phi[n] - 0) * dif01;
        }
        else /* interface Neumann */
        {
          double dif01 = -bc.interfaceValue(x - grad_phi.x*phi[n], y - grad_phi.y*phi[n]);
          q[n] = q1[n] + (-phi[n] - diag) * dif01;
        }
      }

      else if(order==2)
      {
        if(bc.interfaceType()==DIRICHLET)
        {
          double dif01  = (q1[n] - q0[n]) / (diag);
          double dif12  = (q2[n] - q1[n]) / (diag);
          double dif012 = (dif12 - dif01) / (2*diag);
          q[n] = q0[n] + (-phi[n] - 0) * dif01 + (-phi[n] - 0)*(-phi[n] - diag) * dif012;
        }
        else /* interface Neumann */
        {
          double dif01 = (q2[n] - q1[n])/(diag);
          double dif012 = (dif01 + bc.interfaceValue(x - grad_phi.x*phi[n], y - grad_phi.y*phi[n])) / (2*diag);
          q[n] = q1[n] + (-phi[n] - diag) * dif01 + (-phi[n] - diag)*(-phi[n] - 2*diag) * dif012;
        }
      }
    }
  }
  ierr = VecRestoreArray(q_petsc, &q);

  ierr = VecGhostUpdateBegin(q_petsc, INSERT_VALUES, SCATTER_FORWARD);
  ierr = VecGhostUpdateEnd  (q_petsc, INSERT_VALUES, SCATTER_FORWARD);

  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);
}


void my_p4est_level_set::extend_from_interface_to_whole_domain( Vec &phi_petsc, Vec &q_petsc, Vec &q_extended_petsc, int band_to_extend) const
{
  PetscErrorCode ierr;

  double *phi;
  ierr = VecGetArray(phi_petsc, &phi); CHKERRXX(ierr);

  /* find dx and dy smallest */
  p4est_topidx_t tm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t tp = p4est->connectivity->tree_to_vertex[0 + 3];

  double xmin = p4est->connectivity->vertices[3*tm + 0];
  double ymin = p4est->connectivity->vertices[3*tm + 1];
  double xmax = p4est->connectivity->vertices[3*tp + 0];
  double ymax = p4est->connectivity->vertices[3*tp + 1];

  splitting_criteria_t *data = (splitting_criteria_t*) p4est->user_pointer;
  double dx = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  double dy = (ymax-ymin) / pow(2.,(double) data->max_lvl);
  double diag = sqrt(dx*dx + dy*dy);
  InterpolatingFunction interp(p4est, nodes, ghost, myb, ngbd);

  double *q_extended;
  double *q;
  ierr = VecGetArray(q_extended_petsc, &q_extended); CHKERRXX(ierr);
  ierr = VecGetArray(q_petsc, &q); CHKERRXX(ierr);

  /* now buffer the interpolation points */
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    Point2 grad_phi;
    grad_phi.x = (*ngbd)[n].dx_central(phi);
    grad_phi.y = (*ngbd)[n].dy_central(phi);

    if(phi[n]<=0)
      q_extended[n] = q[n];
    else if(phi[n]>0 && phi[n]<band_to_extend*diag && grad_phi.norm_L2()>EPS)
    {
      grad_phi /= grad_phi.norm_L2();
      p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
      p4est_topidx_t tree_id = node->p.piggy3.which_tree;

      p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

      double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];

      double y = int2double_coordinate_transform(node->x) + tree_xmin;
      double x = int2double_coordinate_transform(node->y) + tree_ymin;

      interp.add_point_to_buffer(n, x - grad_phi.x*phi[n], y - grad_phi.y*phi[n]);
    }
    else
      q_extended[n] = 0;
  }
  ierr = VecRestoreArray(phi_petsc, &phi); CHKERRXX(ierr);
  ierr = VecRestoreArray(q_petsc, &q); CHKERRXX(ierr);
  ierr = VecRestoreArray(q_extended_petsc, &q_extended); CHKERRXX(ierr);

  interp.set_input_parameters(q_petsc, quadratic);
  interp.interpolate(q_extended_petsc);

  ierr = VecGhostUpdateBegin(q_extended_petsc, INSERT_VALUES, SCATTER_FORWARD);
  ierr = VecGhostUpdateEnd  (q_extended_petsc, INSERT_VALUES, SCATTER_FORWARD);
}
