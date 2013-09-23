#include "poisson_solver_node_base.h"
#include <src/petsc_compatibility.h>
#include <src/refine_coarsen.h>
#include <src/CASL_math.h>
#include <src/cube2.h>

#ifdef P4_TO_P8
#define MAX_NUM_ELEMENTS_PER_ROW 12 // 3D
#else
#define MAX_NUM_ELEMENTS_PER_ROW 9  // 2D
#endif

#define bc_strength 1.0

PoissonSolverNodeBase::PoissonSolverNodeBase(const my_p4est_node_neighbors_t *node_neighbors, my_p4est_brick_t* myb)
  : node_neighbors_(node_neighbors),
    p4est(node_neighbors->p4est), nodes(node_neighbors->nodes), ghost(node_neighbors->ghost), myb_(myb),
    phi_interp(p4est, nodes, ghost, myb, node_neighbors),
    mu_(1.), diag_add_(0.),
    is_matrix_ready(false), matrix_has_nullspace(false),
    bc_(NULL),
    A(NULL), A_null_space(NULL),
    rhs_(NULL), phi_(NULL), add_(NULL), phi_xx(NULL), phi_yy(NULL)
{
  init();
}

PoissonSolverNodeBase::~PoissonSolverNodeBase()
{
  if (A            != NULL) ierr = MatDestroy(A);                      CHKERRXX(ierr);
  if (A_null_space != NULL) ierr = MatNullSpaceDestroy (A_null_space); CHKERRXX(ierr);
  if (ksp          != NULL) ierr = KSPDestroy(ksp);                    CHKERRXX(ierr);
  if (phi_xx       != NULL) ierr = VecDestroy(phi_xx);                 CHKERRXX(ierr);
  if (phi_yy       != NULL) ierr = VecDestroy(phi_yy);                 CHKERRXX(ierr);
}

void PoissonSolverNodeBase::init()
{
  /*
   * TODO: We can compute the exact number of enteries in the matrix and just
   * allocate that many elements. My guess is its not going to change the memory
   * consumption that much anyway so we might as well allocate for the worst
   * case scenario which is 6 element per row. In places where the grid is
   * uniform we really need 5. In 3D this is 12 vs 7 so its more important ...
   *
   * Also, we only really should allocate 1 per row for points in omega^+ and
   * points for which we use Dirichlet. In the end we are allocating more than
   * we need which may or may not be a real issue in practice ...
   *
   * If we want to do this the correct way, we should first precompute all the
   * weights and probably put them in SparseCRS matrix (CASL) and then construct
   * PETSc matrix such that it uses the same memory space. Note that If copy the
   * stuff its (probably) going to both take longer to execute and consume more
   * memory eventually ...
   *
   * Another simpler approach is to forget about Dirichlet points and also
   * omega^+ domain, but consider the T-junctions and allocate the correct
   * number of elements at least for T-junctions. This does not require
   * precomputation and we only need to chech if a node is T-junction which is
   * much simpler ...
   *
   * We'll see if this becomes a real issue in memory consumption,. My GUESS is
   * it really does not matter in 2D but __might__ be important in 3D for really
   * big problems ...
   */

  // compute global numbering of nodes
  global_node_offset.resize(p4est->mpisize+1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_node_offset[r+1] = global_node_offset[r] + (PetscInt)nodes->global_owned_indeps[r];

  // set up the KSP solver
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  // compute grid parameters
  p4est_topidx_t tm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t tp = p4est->connectivity->tree_to_vertex[0 + 3];

  double xmin = p4est->connectivity->vertices[3*tm + 0];
  double ymin = p4est->connectivity->vertices[3*tm + 1];
  double xmax = p4est->connectivity->vertices[3*tp + 0];
  double ymax = p4est->connectivity->vertices[3*tp + 1];

  dx_min = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  dy_min = (ymax-ymin) / pow(2.,(double) data->max_lvl);
  d_min = MIN(dx_min, dy_min);

  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min);
}

void PoissonSolverNodeBase::preallocate_matrix()
{
  PetscInt num_owned_global = global_node_offset[p4est->mpisize];

  if (A != NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  // set up the matrix
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATMPIAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A,
                     nodes->num_owned_indeps, nodes->num_owned_indeps,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  // preallocate space for matrix
  ierr = MatSeqAIJSetPreallocation(A, MAX_NUM_ELEMENTS_PER_ROW, NULL);
  ierr = MatMPIAIJSetPreallocation(A,
                                   MAX_NUM_ELEMENTS_PER_ROW, NULL,
                                   MAX_NUM_ELEMENTS_PER_ROW, NULL); CHKERRXX(ierr);
}

void PoissonSolverNodeBase::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
#ifdef CASL_THROWS
  if(bc_ == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");

  {
    PetscInt sol_size;
    ierr = VecGetLocalSize(solution, &sol_size); CHKERRXX(ierr);
    if (sol_size != nodes->num_owned_indeps){
      std::ostringstream oss;
      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as num_owned_indeps"
          << "solution.local_size = " << sol_size << " nodes->num_owned_indeps = " << nodes->num_owned_indeps << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
#endif

  // set local add if none was given
  bool local_add = false;
  if(add_ == NULL)
  {
    local_add = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &add_); CHKERRXX(ierr);
    ierr = VecSet(add_, diag_add_); CHKERRXX(ierr);
  }

  // set a local phi if not was given
  bool local_phi = false;
  if(phi_ == NULL)
  {
    local_phi = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi_); CHKERRXX(ierr);
    ierr = VecSet(phi_, -1.); CHKERRXX(ierr);
  }

  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRXX(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRXX(ierr);
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  if (!is_matrix_ready)
  {
    matrix_has_nullspace = true;
    setup_negative_laplace_matrix();

    ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  } else {
    ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER);  CHKERRXX(ierr);
  }

  // setup rhs
  setup_negative_laplace_rhsvec();

  // set the null-space if necessary
  if (matrix_has_nullspace)
    ierr = KSPSetNullSpace(ksp, A_null_space); CHKERRXX(ierr);

  // Solve the system
  ierr = KSPSolve(ksp, rhs_, solution); CHKERRXX(ierr);

  // get rid of local stuff
  if(local_add)
  {
    ierr = VecDestroy(add_); CHKERRXX(ierr);
    add_ = NULL;
  }
  if(local_phi)
  {
    ierr = VecDestroy(phi_); CHKERRXX(ierr);
    phi_ = NULL;
  }
}

void PoissonSolverNodeBase::setup_negative_laplace_matrix()
{
  preallocate_matrix();

  double eps = 1E-6*d_min*d_min;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double *v2q = p4est->connectivity->vertices;

  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p;
  ierr = VecGetArray(phi_,   &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy, &phi_yy_p); CHKERRXX(ierr);
  ierr = VecGetArray(add_,   &add_p   ); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
    p4est_topidx_t tree_it = ni->p.piggy3.which_tree;

    double tree_xmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 0];
    double tree_ymin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 1];

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------

    double x_C  = int2double_coordinate_transform(ni->x) + tree_xmin;
    double y_C  = int2double_coordinate_transform(ni->y) + tree_ymin;

    const quad_neighbor_nodes_of_node_t& qnnn = (*node_neighbors_)[n];

    double dL = qnnn.d_m0;double dR = qnnn.d_p0;double dB = qnnn.d_0m;double dT = qnnn.d_0p;

    double dLB=qnnn.d_m0_m; double dLT=qnnn.d_m0_p; p4est_locidx_t node_LB=qnnn.node_m0_m; p4est_locidx_t node_LT=qnnn.node_m0_p;
    double dRB=qnnn.d_p0_m; double dRT=qnnn.d_p0_p; p4est_locidx_t node_RB=qnnn.node_p0_m; p4est_locidx_t node_RT=qnnn.node_p0_p;
    double dBL=qnnn.d_0m_m; double dBR=qnnn.d_0m_p; p4est_locidx_t node_BL=qnnn.node_0m_m; p4est_locidx_t node_BR=qnnn.node_0m_p;
    double dTL=qnnn.d_0p_m; double dTR=qnnn.d_0p_p; p4est_locidx_t node_TL=qnnn.node_0p_m; p4est_locidx_t node_TR=qnnn.node_0p_p;

    /*
     * global indecies: Note that to insert values into the matrix we need to
     * use global index. Note that although PETSc has a MatSetValuesLocal, that
     * function wont work properly with ghost nodes since the matix does not
     * know the partition of the grid and global indecies for ghost nodes.
     *
     * As a result we compute the ghost indecies manually and insert them using
     * the MatSetValue function.
     *
     * NOTE: Ideally we should be using p4est_gloidx_t for global numbers.
     * However, this requires PetscInt to be 64bit as well otherwise we might
     * run into problems since PETSc internally uses PetscInt for all integer
     * values.
     *
     * As a result, and to prevent weird things from happening, we simpy use
     * PetscInt instead of p4est_gloidx_t for global numbers. This should work
     * for problems that are up to about 2B point big (2^31-1). To go to bigger
     * problems, one should compile PETSc with 64bit support using
     * --with-64-bit-indecies. Please consult PETSc manual for more information.
     *
     * TODO: To get better performance we could first buffer the values in a
     * local SparseCRS matrix and insert them all at once at the end instead of
     * calling MatSetValue every single time. I'm not sure if it will result in
     * much better performance ... to be tested!
     */

    PetscInt node_C_g = petsc_node_gloidx(qnnn.node_00);

    if(is_node_Wall(p4est, ni))
    {
      if(bc_->wallType(x_C,y_C) == DIRICHLET)
      {
        ierr = MatSetValue(A, node_C_g, node_C_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
        if (phi_p[n]<0.) matrix_has_nullspace = false;
        continue;
      }
      if(bc_->wallType(x_C,y_C) == NEUMANN)
      {
        if (is_node_xpWall(p4est, ni)){
          p4est_locidx_t n_L = (dLB == 0) ? node_LB:node_LT;
          PetscInt node_L_g  = petsc_node_gloidx(n_L);

          ierr = MatSetValue(A, node_C_g, node_L_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_C_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
          continue;
        }

        if (is_node_xmWall(p4est, ni)){
          p4est_locidx_t n_R = (dRB == 0) ? node_RB:node_RT;
          PetscInt node_R_g  = petsc_node_gloidx(n_R);

          ierr = MatSetValue(A, node_C_g, node_R_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_C_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

          continue;
        }

        if (is_node_ypWall(p4est, ni)){
          p4est_locidx_t n_B = (dBR == 0) ? node_BR:node_BL;
          PetscInt node_B_g  = petsc_node_gloidx(n_B);

          ierr = MatSetValue(A, node_C_g, node_B_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_C_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

          continue;
        }
        if (is_node_ymWall(p4est, ni)){
          p4est_locidx_t n_T = (dTR == 0) ? node_TR:node_TL;
          PetscInt node_T_g  = petsc_node_gloidx(n_T);

          ierr = MatSetValue(A, node_C_g, node_T_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_C_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

          continue;
        }
      }
    }

    else{
      double phi_C, phi_R, phi_L, phi_B, phi_T;

      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_C, phi_L, phi_R, phi_B, phi_T);

      //---------------------------------------------------------------------
      // interface boundary
      //---------------------------------------------------------------------
      if((ABS(phi_C)<eps && bc_->interfaceType() == DIRICHLET) ){
        ierr = MatSetValue(A, node_C_g, node_C_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);

        matrix_has_nullspace=false;
        continue;
      }

      double Pmhmh = phi_interp(x_C-dx_min/2.,y_C-dy_min/2.);
      double Pmhph = phi_interp(x_C-dx_min/2.,y_C+dy_min/2.);
      double Pphmh = phi_interp(x_C+dx_min/2.,y_C-dy_min/2.);
      double Pphph = phi_interp(x_C+dx_min/2.,y_C+dy_min/2.);

      bool is_ngbd_crossed_neumann = ( Pmhmh*Pmhph<0 || Pmhph*Pphph<0 || Pphph*Pphmh<0 || Pphmh*Pmhmh<0 );

      // far away from the interface
      if(phi_C>0. &&  (!is_ngbd_crossed_neumann || bc_->interfaceType() == DIRICHLET )){
        ierr = MatSetValue(A, node_C_g, node_C_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);

        continue;
      }

      // if far away from the interface or close to it but with dirichlet
      // then finite difference method
      if ( (bc_->interfaceType() == DIRICHLET && phi_C<0.) ||
           (bc_->interfaceType() == NEUMANN   && !is_ngbd_crossed_neumann ) ||
            bc_->interfaceType() == NOINTERFACE)
      {
        double phixx_C = phi_xx_p[n];
        double phiyy_C = phi_yy_p[n];

        bool is_interface_L = (bc_->interfaceType() == DIRICHLET && phi_L*phi_C <= 0.);
        bool is_interface_R = (bc_->interfaceType() == DIRICHLET && phi_R*phi_C <= 0.);
        bool is_interface_T = (bc_->interfaceType() == DIRICHLET && phi_T*phi_C <= 0.);
        bool is_interface_B = (bc_->interfaceType() == DIRICHLET && phi_B*phi_C <= 0.);

        if(  is_interface_B || is_interface_L ||
             is_interface_R || is_interface_T ) matrix_has_nullspace = false;

        // given boundary condition at interface from quadratic interpolation
        if( is_interface_L) {
          double phixx_L = qnnn.f_m0_linear(phi_xx_p);
          double theta_L = interface_Location_With_Second_Order_Derivative(0., dL, phi_C, phi_L, phixx_C, phixx_L);
          if (theta_L<eps) theta_L = eps; if (theta_L>dL) theta_L = dL;
          dLB=0; dLT=0;
          dL = theta_L;
        }
        if( is_interface_R){
          double phixx_R = qnnn.f_p0_linear(phi_xx_p);
          double theta_R = interface_Location_With_Second_Order_Derivative(0., dR, phi_C, phi_R, phixx_C, phixx_R);
          if (theta_R<eps) theta_R = eps; if (theta_R>dR) theta_R = dR;
          dRB=0; dRT=0;
          dR = theta_R;
        }
        if( is_interface_B){
          double phiyy_B = qnnn.f_0m_linear(phi_yy_p);
          double theta_B = interface_Location_With_Second_Order_Derivative(0., dB, phi_C, phi_B, phiyy_C, phiyy_B);
          if (theta_B<eps) theta_B = eps; if (theta_B>dB) theta_B = dB;
          dBL=0; dBR=0;
          dB = theta_B;
        }
        if( is_interface_T){
          double phiyy_T = qnnn.f_0p_linear(phi_yy_p);
          double theta_T = interface_Location_With_Second_Order_Derivative(0., dT, phi_C, phi_T, phiyy_C, phiyy_T);
          if (theta_T<eps) theta_T = eps; if (theta_T>dT) theta_T = dT;
          dTL=0; dTR=0;
          dT = theta_T;
        }

        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        double coeff_L = -2./dL/(dL+dR);
        double coeff_R = -2./dR/(dL+dR);
        double coeff_B = -2./dB/(dB+dT);
        double coeff_T = -2./dT/(dB+dT);

        //---------------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //---------------------------------------------------------------------
        double weight_on_Dyy = 1. - dLT*dLB/dL/(dL+dR) - dRT*dRB/dR/(dL+dR);
        double weight_on_Dxx = 1. - dBL*dBR/dB/(dB+dT) - dTL*dTR/dT/(dB+dT);

        coeff_L *= weight_on_Dxx*mu_;
        coeff_R *= weight_on_Dxx*mu_;
        coeff_B *= weight_on_Dyy*mu_;
        coeff_T *= weight_on_Dyy*mu_;

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------

        double diag = add_p[n]-(coeff_L+coeff_R+coeff_B+coeff_T);
        coeff_L /= diag;
        coeff_R /= diag;
        coeff_B /= diag;
        coeff_T /= diag;

        //---------------------------------------------------------------------
        // addition to diagonal elements
        //---------------------------------------------------------------------
        ierr = MatSetValue(A, node_C_g, node_C_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        if(!is_interface_L) {
          PetscInt node_LT_g = petsc_node_gloidx(node_LT);
          PetscInt node_LB_g = petsc_node_gloidx(node_LB);

          ierr = MatSetValue(A, node_C_g, node_LT_g, coeff_L*dLB/(dLB+dLT), ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_LB_g, coeff_L*dLT/(dLB+dLT), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_R) {
          PetscInt node_RT_g = petsc_node_gloidx(node_RT);
          PetscInt node_RB_g = petsc_node_gloidx(node_RB);

          ierr = MatSetValue(A, node_C_g, node_RT_g, coeff_R*dRB/(dRB+dRT), ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_RB_g, coeff_R*dRT/(dRB+dRT), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_B) {
          PetscInt node_BR_g = petsc_node_gloidx(node_BR);
          PetscInt node_BL_g = petsc_node_gloidx(node_BL);

          ierr = MatSetValue(A, node_C_g, node_BR_g, coeff_B*dBL/(dBL+dBR), ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_BL_g, coeff_B*dBR/(dBL+dBR), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_T) {
          PetscInt node_TR_g = petsc_node_gloidx(node_TR);
          PetscInt node_TL_g = petsc_node_gloidx(node_TL);

          ierr = MatSetValue(A, node_C_g, node_TR_g, coeff_T*dTL/(dTL+dTR), ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_TL_g, coeff_T*dTR/(dTL+dTR), ADD_VALUES); CHKERRXX(ierr);
        }

        if(add_p[n] > 0) matrix_has_nullspace = false;

        continue;
      }

      // if ngbd is crossed and neumman BC
      // then use finite volume method
      // only work if the mesh is uniform close to the interface

      if (is_ngbd_crossed_neumann && bc_->interfaceType() == NEUMANN)
      {
        Cube2 cube;
        cube.x0 = x_C-dx_min/2.;
        cube.x1 = x_C+dx_min/2.;
        cube.y0 = y_C-dy_min/2.;
        cube.y1 = y_C+dy_min/2.;
        QuadValue phi_cube(Pmhmh, Pmhph, Pphmh, Pphph);
        double area_cut_cell = cube.area_In_Negative_Domain(phi_cube);

        if (area_cut_cell>eps*eps)
        {
          p4est_locidx_t quad_mm_idx, quad_pp_idx;
          p4est_topidx_t tree_mm_idx, tree_pp_idx;

          node_neighbors_->find_neighbor_cell_of_node(ni, -1, -1, quad_mm_idx, tree_mm_idx);
          node_neighbors_->find_neighbor_cell_of_node(ni,  1,  1, quad_pp_idx, tree_pp_idx);

          /*
           * z-ordering
           * 0 -> node_mm
           * 1 -> node_pm
           * 2 -> node_mp
           * 3 -> node_pp
           */

          p4est_locidx_t node_L = nodes->local_nodes[P4EST_CHILDREN*quad_mm_idx + 2];
          p4est_locidx_t node_B = nodes->local_nodes[P4EST_CHILDREN*quad_mm_idx + 1];
          p4est_locidx_t node_R = nodes->local_nodes[P4EST_CHILDREN*quad_pp_idx + 1];
          p4est_locidx_t node_T = nodes->local_nodes[P4EST_CHILDREN*quad_pp_idx + 2];

          double fxx,fyy;
          fxx = phi_xx_p[n];
          fyy = phi_yy_p[n];

          double lB = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(Pmhmh, Pphmh, fxx, fxx, dx_min);
          double lT = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(Pmhph, Pphph, fxx, fxx, dx_min);
          double lL = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(Pmhmh, Pmhph, fyy, fyy, dy_min);
          double lR = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(Pphmh, Pphph, fyy, fyy, dy_min);

          double UL = -lL/dx_min;
          double UR = -lR/dx_min;
          double UT = -lT/dy_min;
          double UB = -lB/dy_min;

          UL *= mu_;
          UR *= mu_;
          UT *= mu_;
          UB *= mu_;

          double U = add_p[n]*area_cut_cell-UR-UL-UB-UT;

          UL/=U;
          UR/=U;
          UB/=U;
          UT/=U;

          PetscInt node_R_g = petsc_node_gloidx(node_R);
          PetscInt node_L_g = petsc_node_gloidx(node_L);
          PetscInt node_T_g = petsc_node_gloidx(node_T);
          PetscInt node_B_g = petsc_node_gloidx(node_B);

          ierr = MatSetValue(A, node_C_g, node_C_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_R_g, UR,  ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_L_g, UL,  ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_B_g, UB,  ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_C_g, node_T_g, UT,  ADD_VALUES); CHKERRXX(ierr);

          if(add_p[n] > 0) matrix_has_nullspace = false;
        }
        else{
          ierr = MatSetValue(A, node_C_g, node_C_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
        }
        continue;
      }
    }
  }

  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

  // restore pointers
  ierr = VecRestoreArray(phi_,   &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy, &phi_yy_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(add_,   &add_p   ); CHKERRXX(ierr);

  // check for null space
  if (matrix_has_nullspace)
  {
    if (A_null_space == NULL) // pun not intended!
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, PETSC_NULL, &A_null_space); CHKERRXX(ierr);

    ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
  }
}

void PoissonSolverNodeBase::setup_negative_laplace_rhsvec()
{

  double eps=1E-6*d_min*d_min;

  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double *v2q = p4est->connectivity->vertices;

  double *phi_p, *phi_xx_p, *phi_yy_p, *rhs_p, *add_p;
  ierr = VecGetArray(phi_,   &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy, &phi_yy_p); CHKERRXX(ierr);
  ierr = VecGetArray(rhs_,   &rhs_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(add_,   &add_p   ); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n+nodes->offset_owned_indeps);
    p4est_topidx_t tree_it = ni->p.piggy3.which_tree;

    double tree_xmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 0];
    double tree_ymin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 1];

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------

    double x_C  = int2double_coordinate_transform(ni->x) + tree_xmin;
    double y_C  = int2double_coordinate_transform(ni->y) + tree_ymin;

    const quad_neighbor_nodes_of_node_t& qnnn = (*node_neighbors_)[n];

    double dL = qnnn.d_m0;double dR = qnnn.d_p0;double dB = qnnn.d_0m;double dT = qnnn.d_0p;

    double dLB = qnnn.d_m0_m; double dLT = qnnn.d_m0_p;
    double dRB = qnnn.d_p0_m; double dRT = qnnn.d_p0_p;
    double dBL = qnnn.d_0m_m; double dBR = qnnn.d_0m_p;
    double dTL = qnnn.d_0p_m; double dTR = qnnn.d_0p_p;

    if(is_node_Wall(p4est, ni)){
      if(bc_->wallType(x_C,y_C) == DIRICHLET){
        rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C);
        continue;
      }
      if(bc_->wallType(x_C,y_C) == NEUMANN){

        if (is_node_xpWall(p4est, ni)){
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*dL;
          continue;
        }

        if (is_node_xmWall(p4est, ni)){
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*dR;
          continue;
        }

        if (is_node_ypWall(p4est, ni)){
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*dB;
          continue;
        }
        if (is_node_ymWall(p4est, ni)){
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*dT;
          continue;
        }
      }
    }

    double phi_C, phi_R, phi_L, phi_B, phi_T;
    qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_C, phi_L, phi_R, phi_B, phi_T);
    //---------------------------------------------------------------------
    // interface boundary
    //---------------------------------------------------------------------
    if((ABS(phi_C)<eps && bc_->interfaceType() == DIRICHLET) ){
      rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C);
      continue;
    }

    double Pmhph = phi_interp(x_C-dx_min/2,y_C+dy_min/2);
    double Pmhmh = phi_interp(x_C-dx_min/2,y_C-dy_min/2);
    double Pphmh = phi_interp(x_C+dx_min/2,y_C-dy_min/2);
    double Pphph = phi_interp(x_C+dx_min/2,y_C+dy_min/2);

    bool is_ngbd_crossed_neumann = ( Pmhmh*Pmhph<0 || Pmhph*Pphph<0 || Pphph*Pphmh<0 || Pphmh*Pmhmh<0 );

    // far away from the interface
    if (phi_C>eps  && (!is_ngbd_crossed_neumann || bc_->interfaceType() == DIRICHLET )){
      if(bc_->interfaceType()==DIRICHLET)
        rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C);
      else
        rhs_p[n] = 0;
      continue;
    }

    // if far away from the interface or close to the interface but inside and with dirichlet boundary condition
    // then finite difference method
    //        if (phi_C<-eps && (!is_ngbd_crossed || bc_->interfaceType()==DIRICHLET ))
    if ( (bc_->interfaceType()  == DIRICHLET && phi_C<0.) ||
         (bc_->interfaceType()  == NEUMANN   && !is_ngbd_crossed_neumann ) ||
          bc_->interfaceType()  == NOINTERFACE)
    {
      double phixx_C = phi_xx_p[n];
      double phiyy_C = phi_yy_p[n];

      bool is_interface_L = (bc_->interfaceType() == DIRICHLET && phi_L*phi_C <= 0.);
      bool is_interface_R = (bc_->interfaceType() == DIRICHLET && phi_R*phi_C <= 0.);
      bool is_interface_T = (bc_->interfaceType() == DIRICHLET && phi_T*phi_C <= 0.);
      bool is_interface_B = (bc_->interfaceType() == DIRICHLET && phi_B*phi_C <= 0.);

      double val_interface_L = 0.;
      double val_interface_R = 0.;
      double val_interface_B = 0.;
      double val_interface_T = 0.;

      // given boundary condition at interface from quadratic interpolation
      if( is_interface_L) {
        double phixx_L = qnnn.f_m0_linear(phi_xx_p);
        double theta_L = interface_Location_With_Second_Order_Derivative(0., dL, phi_C, phi_L,phixx_C,phixx_L);
        if (theta_L<eps) theta_L = eps; if (theta_L>dL) theta_L = dL;
        dLB=0; dLT=0;
        val_interface_L = bc_->interfaceValue(x_C - theta_L, y_C);
        dL = theta_L;
      }
      if( is_interface_R){
        double phixx_R = qnnn.f_p0_linear(phi_xx_p);
        double theta_R = interface_Location_With_Second_Order_Derivative(0., dR, phi_C, phi_R,phixx_C,phixx_R);
        if (theta_R<eps) theta_R = eps; if (theta_R>dR) theta_R = dR;
        dRB=0; dRT=0;
        val_interface_R = bc_->interfaceValue(x_C + theta_R, y_C);
        dR = theta_R;
      }
      if( is_interface_B){
        double phiyy_B = qnnn.f_0m_linear(phi_yy_p);
        double theta_B = interface_Location_With_Second_Order_Derivative(0., dB, phi_C, phi_B,phiyy_C,phiyy_B);
        if (theta_B<eps) theta_B = eps; if (theta_B>dB) theta_B = dB;
        dBL=0; dBR=0;
        val_interface_B = bc_->interfaceValue(x_C, y_C - theta_B);
        dB = theta_B;
      }
      if( is_interface_T){
        double phiyy_T = qnnn.f_0p_linear(phi_yy_p);
        double theta_T = interface_Location_With_Second_Order_Derivative(0., dT, phi_C, phi_T,phiyy_C,phiyy_T);
        if (theta_T<eps) theta_T = eps; if (theta_T>dT) theta_T = dT;
        dTL=0; dTR=0;
        val_interface_T = bc_->interfaceValue(x_C, y_C + theta_T);
        dT = theta_T;
      }

      //---------------------------------------------------------------------
      // Shortley-Weller method, dimension by dimension
      //---------------------------------------------------------------------
      double coeff_L = -2/dL/(dL+dR);
      double coeff_R = -2/dR/(dL+dR);
      double coeff_B = -2/dB/(dB+dT);
      double coeff_T = -2/dT/(dB+dT);

      //---------------------------------------------------------------------
      // compensating the error of linear interpolation at T-junction using
      // the derivative in the transversal direction
      //---------------------------------------------------------------------
      double weight_on_Dyy = 1 - dLT*dLB/dL/(dL+dR) - dRT*dRB/dR/(dL+dR);
      double weight_on_Dxx = 1 - dBL*dBR/dB/(dB+dT) - dTL*dTR/dT/(dB+dT);

      coeff_L *= weight_on_Dxx*mu_;
      coeff_R *= weight_on_Dxx*mu_;
      coeff_B *= weight_on_Dyy*mu_;
      coeff_T *= weight_on_Dyy*mu_;

      //---------------------------------------------------------------------
      // diag scaling
      //---------------------------------------------------------------------

      double diag = add_p[n]-(coeff_L+coeff_R+coeff_B+coeff_T);
      coeff_L  /= diag;
      coeff_R  /= diag;
      coeff_B  /= diag;
      coeff_T  /= diag;
      rhs_p[n] /= diag;

      if(is_interface_L) rhs_p[n] -= coeff_L*val_interface_L;
      if(is_interface_R) rhs_p[n] -= coeff_R*val_interface_R;
      if(is_interface_B) rhs_p[n] -= coeff_B*val_interface_B;
      if(is_interface_T) rhs_p[n] -= coeff_T*val_interface_T;

      continue;
    }

    // if ngbd is crossed and neumman BC
    // then use finite volume method
    // only work if the mesh is uniform close to the interface

    if (is_ngbd_crossed_neumann && bc_->interfaceType()==NEUMANN)
    {
      Cube2 cube;
      cube.x0 = x_C-dx_min/2.;
      cube.x1 = x_C+dx_min/2.;
      cube.y0 = y_C-dy_min/2.;
      cube.y1 = y_C+dy_min/2.;
      QuadValue phi_cube(Pmhmh,Pmhph,Pphmh,Pphph);
      double area_cut_cell = cube.area_In_Negative_Domain(phi_cube);

      if (area_cut_cell>eps*eps){

        double fxx,fyy;
        fxx = phi_xx_p[n];
        fyy = phi_yy_p[n];

        double lB = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(Pmhmh, Pphmh, fxx, fxx, dx_min);
        double lT = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(Pmhph, Pphph, fxx, fxx, dx_min);
        double lL = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(Pmhmh, Pmhph, fyy, fyy, dy_min);
        double lR = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(Pphmh, Pphph, fyy, fyy, dy_min);

        double UL =-lL/dx_min;
        double UR =-lR/dx_min;
        double UT =-lT/dy_min;
        double UB =-lB/dy_min;

        UL *= mu_;
        UR *= mu_;
        UT *= mu_;
        UB *= mu_;

        double U = add_p[n]*area_cut_cell-UR-UL-UB-UT;

        QuadValue bc_value(     bc_->interfaceValue(cube.x0,cube.y0),
                                bc_->interfaceValue(cube.x0,cube.y1),
                                bc_->interfaceValue(cube.x1,cube.y0),
                                bc_->interfaceValue(cube.x1,cube.y1) );

        double integral_bc = cube.integrate_Over_Interface(bc_value,phi_cube);
        rhs_p[n] *= area_cut_cell;
        rhs_p[n] += mu_*integral_bc;
        rhs_p[n] /= U;

      }
      else{
        rhs_p[n] = 0.;
      }
      continue;
    }
  }

  // restore the pointers
  ierr = VecRestoreArray(phi_,   &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy, &phi_yy_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(add_,   &add_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_,   &phi_p   ); CHKERRXX(ierr);

  if (matrix_has_nullspace)
    ierr = MatNullSpaceRemove(A_null_space, rhs_, NULL); CHKERRXX(ierr);
}

void PoissonSolverNodeBase::set_phi(Vec phi)
{
  phi_ = phi;

  // Allocate memory for second derivaties
  ierr = VecCreateGhost(p4est, nodes, &phi_xx); CHKERRXX(ierr);
  ierr = VecDuplicate(phi_xx, &phi_yy); CHKERRXX(ierr);

  // Access internal data
  double *phi_p, *phi_xx_p, *phi_yy_p;
  ierr = VecGetArray(phi_,   &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy, &phi_yy_p); CHKERRXX(ierr);

  // Compute phi_xx on local nodes
  for (p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    phi_xx_p[n] = (*node_neighbors_)[n].dxx_central(phi_p);

  // Send ghost values for phi_xx
  ierr = VecGhostUpdateBegin(phi_xx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // Compute phi_yy on local nodes
  for (p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
    phi_yy_p[n] = (*node_neighbors_)[n].dyy_central(phi_p);

  // receive the ghost values for phi_xx
  ierr = VecGhostUpdateEnd(phi_xx, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx, &phi_xx_p); CHKERRXX(ierr);

  // Send ghost values for Fyy and receive them
  ierr = VecGhostUpdateBegin(phi_yy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(phi_yy, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // restore Fyy array
  ierr = VecRestoreArray(phi_yy, &phi_yy_p); CHKERRXX(ierr);

  // restore input_vec_ array
  ierr = VecRestoreArray(phi_, &phi_p); CHKERRXX(ierr);

  phi_interp.set_input_parameters(phi_, quadratic_non_oscillatory, phi_xx, phi_yy);
}
