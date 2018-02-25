#include <src/my_p4est_poisson_jump_nodes_voronoi.h>
#include <src/my_p4est_refine_coarsen.h>

#include <algorithm>

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_PoissonSolverNodeBasedJump_matrix_preallocation;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_setup_linear_system;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_rhsvec_setup;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_KSPSolve;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_solve;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_compute_voronoi_points;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_compute_voronoi_cell;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_interpolate_to_tree;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bc_strength 1.0

my_p4est_poisson_jump_nodes_voronoi_t::my_p4est_poisson_jump_nodes_voronoi_t(const my_p4est_node_neighbors_t *node_neighbors,
                                                     const my_p4est_cell_neighbors_t *cell_neighbors)
  : ngbd_n(node_neighbors),  ngbd_c(cell_neighbors), myb(node_neighbors->myb),
    p4est(node_neighbors->p4est), ghost(node_neighbors->ghost), nodes(node_neighbors->nodes),
    phi(NULL), rhs(NULL), sol_voro(NULL),
    voro_global_offset(p4est->mpisize),
    interp_phi(node_neighbors),
    rhs_m(node_neighbors),
    rhs_p(node_neighbors),
    local_mu(false), local_add(false),
    local_u_jump(false), local_mu_grad_u_jump(false),
    mu_m(&mu_constant_m), mu_p(&mu_constant_p),
    add_m(&add_constant_m), add_p(&add_constant_p),
    u_jump(&zero), mu_grad_u_jump(&zero),
    A(PETSC_NULL), A_null_space(PETSC_NULL), ksp(PETSC_NULL),
    is_voronoi_partition_constructed(false), is_matrix_computed(false), matrix_has_nullspace(false)
{
  // set up the KSP solver
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);

  xyz_min_max(p4est, xyz_min, xyz_max);

  dxyz_min(p4est, dxyz_min_);

#ifdef P4_TO_P8
  d_min = MIN(dxyz_min_[0],dxyz_min_[1],dxyz_min_[2]);
  diag_min = sqrt(SQR(dxyz_min_[0]) + SQR(dxyz_min_[1]) + SQR(dxyz_min_[2]));
#else
  d_min = MIN(dxyz_min_[0],dxyz_min_[1]);
  diag_min = sqrt(SQR(dxyz_min_[0]) + SQR(dxyz_min_[1]));
#endif
}

my_p4est_poisson_jump_nodes_voronoi_t::~my_p4est_poisson_jump_nodes_voronoi_t()
{
  if(A            != PETSC_NULL) { ierr = MatDestroy(A);                      CHKERRXX(ierr); }
  if(A_null_space != PETSC_NULL) { ierr = MatNullSpaceDestroy(A_null_space);  CHKERRXX(ierr); }
  if(ksp          != PETSC_NULL) { ierr = KSPDestroy(ksp);                    CHKERRXX(ierr); }
  if(rhs          != PETSC_NULL) { ierr = VecDestroy(rhs);                    CHKERRXX(ierr); }
  if(local_mu)             { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_m); delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_p); }
  if(local_add)            { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(add_m); delete dynamic_cast<my_p4est_interpolation_nodes_t*>(add_p); }
  if(local_u_jump)         { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(u_jump); }
  if(local_mu_grad_u_jump) { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_grad_u_jump); }
}

PetscErrorCode my_p4est_poisson_jump_nodes_voronoi_t::VecCreateGhostVoronoiRhs()
{
  PetscErrorCode ierr = 0;
  PetscInt num_local = num_local_voro;
  PetscInt num_global = voro_global_offset[p4est->mpisize];

  std::vector<PetscInt> ghost_voro(voro_seeds.size() - num_local, 0);

  for (size_t i = 0; i<ghost_voro.size(); ++i)
  {
    ghost_voro[i] = voro_ghost_local_num[i] + voro_global_offset[voro_ghost_rank[i]];
  }

  if(rhs!=PETSC_NULL) VecDestroy(rhs);

  ierr = VecCreateGhost(p4est->mpicomm, num_local_voro, num_global,
                        ghost_voro.size(), (const PetscInt*)&ghost_voro[0], &rhs); CHKERRQ(ierr);
  ierr = VecSetFromOptions(rhs); CHKERRQ(ierr);

  return ierr;
}

void my_p4est_poisson_jump_nodes_voronoi_t::set_phi(Vec phi)
{
  this->phi = phi;
  interp_phi.set_input(phi, linear);
}

void my_p4est_poisson_jump_nodes_voronoi_t::set_rhs(Vec rhs_m, Vec rhs_p)
{
  this->rhs_m.set_input(rhs_m, linear);
  this->rhs_p.set_input(rhs_p, linear);
}

void my_p4est_poisson_jump_nodes_voronoi_t::set_diagonals(double add_m_, double add_p_)
{
  if(local_add)
  {
    delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->add_m);
    delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->add_p);
    local_add = false;
  }
  add_constant_m.set(add_m_);
  this->add_m = &add_constant_m;
  add_constant_p.set(add_p_);
  this->add_p = &add_constant_p;
}

void my_p4est_poisson_jump_nodes_voronoi_t::set_diagonals(Vec add_m_, Vec add_p_)
{
  if(local_add)
  {
    delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->add_m);
    delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->add_p);
  }
  my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
  tmp->set_input(add_m_, linear);
  this->add_m = tmp;
  tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
  tmp->set_input(add_p_, linear);
  this->add_p = tmp;
  local_add = true;
}

#ifdef P4_TO_P8
void my_p4est_poisson_jump_nodes_voronoi_t::set_bc(BoundaryConditions3D& bc)
#else
void my_p4est_poisson_jump_nodes_voronoi_t::set_bc(BoundaryConditions2D& bc)
#endif
{
  this->bc = &bc;
  is_matrix_computed = false;
}

void my_p4est_poisson_jump_nodes_voronoi_t::set_mu(double mu_m_, double mu_p_)
{
  if(local_mu)
  {
    delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_m);
    delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_p);
    local_mu = false;
  }
  mu_constant_m.set(mu_m_);
  mu_m = &mu_constant_m;
  mu_constant_p.set(mu_p_);
  mu_p = &mu_constant_p;
}

void my_p4est_poisson_jump_nodes_voronoi_t::set_mu(Vec mu_m, Vec mu_p)
{
  if(local_mu)
  {
    delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->mu_m);
    delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->mu_p);
  }
  my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
  tmp->set_input(mu_m, linear);
  this->mu_m = tmp;

  tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
  tmp->set_input(mu_p, linear);
  this->mu_p = tmp;
  local_mu = true;
}

void my_p4est_poisson_jump_nodes_voronoi_t::set_u_jump(Vec u_jump)
{
  if(local_u_jump) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->u_jump);
  my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
  tmp->set_input(u_jump, linear);
  this->u_jump = tmp;
  local_u_jump = true;
}

void my_p4est_poisson_jump_nodes_voronoi_t::set_mu_grad_u_jump(Vec mu_grad_u_jump)
{
  if(local_mu_grad_u_jump) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->mu_grad_u_jump);
  my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
  tmp->set_input(mu_grad_u_jump, linear);
  this->mu_grad_u_jump = tmp;
  local_mu_grad_u_jump = true;
}

void my_p4est_poisson_jump_nodes_voronoi_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_solve, A, rhs, ksp, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  if(bc == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");

  PetscInt sol_size;
  ierr = VecGetLocalSize(solution, &sol_size); CHKERRXX(ierr);
  if (sol_size != nodes->num_owned_indeps){
    std::ostringstream oss;
    oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as num_owned_indeps"
        << "solution.local_size = " << sol_size << " nodes->num_owned_indeps = " << nodes->num_owned_indeps << std::endl;
    throw std::invalid_argument(oss.str());
  }
#endif

  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);
  // [Raphael]: is this really relevant since sol_voro is build by duplication of rhs (no copy)
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  /* first compute the voronoi partition */
  if(!is_voronoi_partition_constructed)
  {
    compute_voronoi_points();
    is_voronoi_partition_constructed = true;
  }

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  if(!is_matrix_computed)
  {
    matrix_has_nullspace = true;

//    ierr = PetscPrintf(p4est->mpicomm, "Assembling linear system ...\n"); CHKERRXX(ierr);
    setup_linear_system();
//    ierr = PetscPrintf(p4est->mpicomm, "Done assembling linear system.\n"); CHKERRXX(ierr);

    is_matrix_computed = true;
    ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  } else {
    setup_negative_laplace_rhsvec();
    ierr = KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER);  CHKERRXX(ierr);
  }

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRXX(ierr);
  ierr = PCSetType(pc, pc_type); CHKERRXX(ierr);

  /* If using hypre, we can make some adjustments here. The most important parameters to be set are:
   * 1- Strong Threshold
   * 2- Coarsennig Type
   * 3- Truncation Factor
   *
   * Plerase refer to HYPRE manual for more information on the actual importance or check Mohammad Mirzadeh's
   * summary of HYPRE papers! Also for a complete list of all the options that can be set from PETSc, one can
   * consult the 'src/ksp/pc/impls/hypre.c' in the PETSc home directory.
   */
  if (!strcmp(pc_type, PCHYPRE)){
    /* 1- Strong threshold:
     * Between 0 to 1
     * "0 "gives better convergence rate (in 3D).
     * Suggested values (By Hypre manual): 0.25 for 2D, 0.5 for 3D
    */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_strong_threshold", "0.5"); CHKERRXX(ierr);

    /* 2- Coarsening type
     * Available Options:
     * "CLJP","Ruge-Stueben","modifiedRuge-Stueben","Falgout", "PMIS", "HMIS". Falgout is usually the best.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_coarsen_type", "Falgout"); CHKERRXX(ierr);

    /* 3- Trancation factor
     * Greater than zero.
     * Use zero for the best convergence. However, if you have memory problems, use greate than zero to save some memory.
     */
    ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_truncfactor", "0.1"); CHKERRXX(ierr);
    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
    if (matrix_has_nullspace){
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
    }
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  /* set the nullspace */
  if (matrix_has_nullspace){
    // PETSc removed the KSPSetNullSpace in 3.6.0 ... Use MatSetNullSpace instead
#if PETSC_VERSION_GE(3,6,0)
    ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
//    ierr = MatSetTransposeNullSpace(A, A_null_space); CHKERRXX(ierr);
#else
    ierr = KSPSetNullSpace(ksp, A_null_space); CHKERRXX(ierr);
#endif
  }

  /* Solve the system */
  ierr = VecDuplicate(rhs, &sol_voro); CHKERRXX(ierr);

//  ierr = PetscPrintf(p4est->mpicomm, "Solving linear system ...\n"); CHKERRXX(ierr);
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_KSPSolve, ksp, rhs, sol_voro, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, sol_voro); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_PoissonSolverNodeBasedJump_KSPSolve, ksp, rhs, sol_voro, 0); CHKERRXX(ierr);
//  ierr = PetscPrintf(p4est->mpicomm, "Done solving linear system.\n"); CHKERRXX(ierr);

  double res;
  ierr = KSPGetResidualNorm(ksp, &res);
  //ierr = PetscPrintf(p4est->mpicomm, "Done solving linear system ... residual = %f\n", res); CHKERRXX(ierr);

  /* update ghosts */
  ierr = VecGhostUpdateBegin(sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* interpolate the solution back onto the original mesh */
  interpolate_solution_from_voronoi_to_tree(solution);

  ierr = VecDestroy(sol_voro); sol_voro = NULL; CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_jump_nodes_voronoi_t::compute_voronoi_points()
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_compute_voronoi_points, 0, 0, 0, 0); CHKERRXX(ierr);

  // clear data first if needed
  if(grid2voro.size()!=0)
  {
    for(unsigned int n=0; n<grid2voro.size(); ++n)
      grid2voro[n].clear();
  }
  grid2voro.resize(nodes->indep_nodes.elem_count);

  voro_seeds.clear();

  // here we start
  int mpiret;
  const double *phi_read_only_p;
  ierr = VecGetArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr);

  std::vector<p4est_locidx_t> marked_nodes;


  /* --- 0. INITIALIZATION OF THE PROCEDURE ---*/
  /* Find the projected points associated with shared grid nodes, first. If a layer grid node is close to the
   * interface, the projected point is calculated and shared with the sharer process(es). After communication
   * of these "shared" projected points, every process will be able to prevent the interface to be sampled with
   * points that are "too close" to each other (less than diag_min/close_distance_factor)
   * The goal here is to provide processes with sufficient information to avoid creating two projected points
   * that are two close to each other, by initializing so the projected points associated with ghost layers
   * across process boundaries in a globally consistent way. Then every local projected point (projection of
   * a grid node interior to the process region) is added to the set of interface sampling points if it is
   * not too close to any other point already in the set of sampling points.
   * The grid nodes that are far from the interface or that need to be added to the list of Voronoi seeds for
   * some other reason are added to voro_seeds.
   */
  //
  // From now on, marked_nodes contains local indices of grid nodes that need to be added to voro_seeds
  //
  /* lists of projected points created from layer nodes to be shared with (an)other process(es) */
  std::vector< std::vector<projected_point_t> > buff_shared_projected_points_send(p4est->mpisize);
  /* lists of projected points created from layer nodes by (an)other process(es) to be shared with the current process */
  std::vector< std::vector<projected_point_t> > buff_shared_projected_points_recv(p4est->mpisize);
  std::vector<int> send_projected_to(p4est->mpisize, 0);

  for(size_t l=0; l<ngbd_n->get_layer_size(); ++l)
  {
    /* get the shared node */
    p4est_locidx_t n = ngbd_n->get_layer_node(l);
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    /* check if it is close to interface */
#ifdef P4_TO_P8
    double p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p;
    (*ngbd_n).get_neighbors(n).ngbd_with_quadratic_interpolation(phi_read_only_p, p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p);
    if(p_000*p_m00<=0 || p_000*p_p00<=0 || p_000*p_0m0<=0 || p_000*p_0p0<=0 || p_000*p_00m<=0 || p_000*p_00p<=0)
#else
    double p_00, p_m0, p_p0, p_0m, p_0p;
    (*ngbd_n).get_neighbors(n).ngbd_with_quadratic_interpolation(phi_read_only_p, p_00, p_m0, p_p0, p_0m, p_0p);
    if(p_00*p_m0<=0 || p_00*p_p0<=0 || p_00*p_0m<=0 || p_00*p_0p<=0)
#endif
    {
      if(is_node_Wall(p4est, node))
        marked_nodes.push_back(n); // when the interface is close to the boundary, we add the node to the seeds as well

      /* get the sharer processes, fill the buffers and initialize the future communications */
      size_t num_sharers = (size_t) node->pad8;
      sc_recycle_array_t *rec = (sc_recycle_array_t*)sc_array_index(&nodes->shared_indeps, num_sharers - 1);
      int *sharers;
      size_t sharers_index;
      if(nodes->shared_offsets == NULL)
      {
        P4EST_ASSERT(node->pad16 >= 0);
        sharers_index = (size_t) node->pad16;
      }
      else
      {
        P4EST_ASSERT(node->pad16 == -1);
        sharers_index = (size_t) nodes->shared_offsets[n];
      }

      sharers = (int*) sc_array_index(&rec->a, sharers_index);
      for(size_t s=0; s<num_sharers; ++s)
        send_projected_to[sharers[s]] = 1;

      /* calculate the projected point */
      double d = phi_read_only_p[n];
#ifdef P4_TO_P8
      Point3 dp((*ngbd_n).get_neighbors(n).dx_central(phi_read_only_p), (*ngbd_n).get_neighbors(n).dy_central(phi_read_only_p), (*ngbd_n).get_neighbors(n).dz_central(phi_read_only_p));
#else
      Point2 dp((*ngbd_n).get_neighbors(n).dx_central(phi_read_only_p), (*ngbd_n).get_neighbors(n).dy_central(phi_read_only_p));
#endif
      // note: if phi is somehow a signed distance, dp has no dimension
      if(dp.norm_L2() > EPS) // no issue to calculate the projected point, so do it
      {
        dp /= dp.norm_L2();
        double xn = node_x_fr_n(n, p4est, nodes);
        double yn = node_y_fr_n(n, p4est, nodes);
        projected_point_t projected_point_n;
        projected_point_n.x = xn-d*dp.x;
        projected_point_n.y = yn-d*dp.y;
        projected_point_n.dx = dp.x;
        projected_point_n.dy = dp.y;
#ifdef P4_TO_P8
        double zn = node_z_fr_n(n, p4est, nodes);
        projected_point_n.z = zn-d*dp.z;
        projected_point_n.dz = dp.z;
#endif

        // add the projected point to the buffers to be sent to all appropriate sharer processes
        for(size_t s=0; s<num_sharers; ++s)
          buff_shared_projected_points_send[sharers[s]].push_back(projected_point_n);
        // fill the self-communication recv buffer
        buff_shared_projected_points_recv[p4est->mpirank].push_back(projected_point_n);
      }
      else // the projected point can't be calculated in a reliable way, the gradient is ill-defined (e.g. center of an under-resolved sphere)
        marked_nodes.push_back(n); // then, we add the grid node to the seeds as well
    }
    else
      marked_nodes.push_back(n); // the point is far from the interface so add it to the seeds
  }
  /* add the grid nodes that are voronoi seeds (not close to the interface, or for some other reason) */
  /* layer nodes (marked here above) first */
  for(unsigned int i=0; i<marked_nodes.size(); ++i)
  {
    p4est_locidx_t n = marked_nodes[i];
    double xn = node_x_fr_n(n, p4est, nodes);
    double yn = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double zn = node_z_fr_n(n, p4est, nodes);
    Point3 p(xn, yn, zn);
#else
    Point2 p(xn, yn);
#endif
    grid2voro[n].push_back(voro_seeds.size());
    voro_seeds.push_back(p);
  }
  // RESET marked_nodes:
  marked_nodes.clear();
  //
  // From now on, marked_nodes will be set to contain indices of local nodes whose projection needs to be calculated
  //

  /* send the projected points to the corresponding neighbor processes, non-blocking */
  std::vector<MPI_Request> req_shared_projected_points;
  for(int r=0; r<p4est->mpisize; ++r)
  {
    if(send_projected_to[r]==1)
    {
      MPI_Request req;
      mpiret = MPI_Isend(&buff_shared_projected_points_send[r][0], buff_shared_projected_points_send[r].size()*sizeof(projected_point_t), MPI_BYTE, r, 4, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
      req_shared_projected_points.push_back(req);
    }
  }
  /* loop through local (inner) nodes, add the relevant ones to the voronoi seeds and mark the other ones */
  for (size_t l = 0; l < ngbd_n->get_local_size(); ++l) {
    p4est_locidx_t n = ngbd_n->get_local_node(l);
#ifdef P4_TO_P8
    double p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p;
    (*ngbd_n).get_neighbors(n).ngbd_with_quadratic_interpolation(phi_read_only_p, p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p);
    if(!(p_000*p_m00<=0 || p_000*p_p00<=0 || p_000*p_0m0<=0 || p_000*p_0p0<=0 || p_000*p_00m<=0 || p_000*p_00p<=0))
#else
    double p_00, p_m0, p_p0, p_0m, p_0p;
    (*ngbd_n).get_neighbors(n).ngbd_with_quadratic_interpolation(phi_read_only_p, p_00, p_m0, p_p0, p_0m, p_0p);
    if(!(p_00*p_m0<=0 || p_00*p_p0<=0 || p_00*p_0m<=0 || p_00*p_0p<=0))
#endif
    {
      // not close to the interface, so add it to the seeds
      double xn = node_x_fr_n(n, p4est, nodes);
      double yn = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double zn = node_z_fr_n(n, p4est, nodes);
      Point3 p(xn, yn, zn);
#else
      Point2 p(xn, yn);
#endif
      grid2voro[n].push_back(voro_seeds.size());
      voro_seeds.push_back(p);
    }
    else
    {
      // the grid node is close to the interface
      if(is_node_Wall(p4est, ((p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n))))
      {
        // once again, if it's a wall node, we add it to the seeds as well
        double xn = node_x_fr_n(n, p4est, nodes);
        double yn = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double zn = node_z_fr_n(n, p4est, nodes);
        Point3 p(xn, yn, zn);
#else
        Point2 p(xn, yn);
#endif
        grid2voro[n].push_back(voro_seeds.size());
        voro_seeds.push_back(p);
      }
#ifdef P4_TO_P8
      Point3 dp((*ngbd_n).get_neighbors(n).dx_central(phi_read_only_p), (*ngbd_n).get_neighbors(n).dy_central(phi_read_only_p), (*ngbd_n).get_neighbors(n).dz_central(phi_read_only_p));
#else
      Point2 dp((*ngbd_n).get_neighbors(n).dx_central(phi_read_only_p), (*ngbd_n).get_neighbors(n).dy_central(phi_read_only_p));
#endif
      if(dp.norm_L2() > EPS)
        marked_nodes.push_back(n); // the projection can be safely calculated, add it to the marked nodes
      else
      {
        // under-resolved case, the projection can't be calculated reliably (e.g. center of an under-resolved
        // sphere), add the grid node to the seeds
        double xn = node_x_fr_n(n, p4est, nodes);
        double yn = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
        double zn = node_z_fr_n(n, p4est, nodes);
        Point3 p(xn, yn, zn);
#else
        Point2 p(xn, yn);
#endif
        grid2voro[n].push_back(voro_seeds.size());
        voro_seeds.push_back(p);
      }
    }
  }
  /* compute how many messages (buffers of "ghost" projected points) we are expecting to receive */
  int nb_rcv;
  std::vector<int> nb_int_per_proc(p4est->mpisize, 1);
  mpiret = MPI_Reduce_scatter(&send_projected_to[0], &nb_rcv, &nb_int_per_proc[0], MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  /* Receive the buffers */
  while(nb_rcv > 0)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 4, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);
    int vec_size;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &vec_size); SC_CHECK_MPI(mpiret);
    vec_size /= sizeof(projected_point_t);

    buff_shared_projected_points_recv[status.MPI_SOURCE].resize(vec_size);
    mpiret = MPI_Recv(&buff_shared_projected_points_recv[status.MPI_SOURCE][0], vec_size*sizeof(projected_point_t), MPI_BYTE, status.MPI_SOURCE, 4, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);

    nb_rcv--;
  }

  /* Create the list of projected points in a globally consistent way */
#ifdef P4_TO_P8
  std::vector<Point3> projected_points;
  std::vector<Point3> projected_points_grad;
#else
  std::vector<Point2> projected_points;
  std::vector<Point2> projected_points_grad;
#endif
  /* initialize the list of projected points with the "ghost" projected points and the projected layer points.
   * Note the loop over increasing rank processes to make the initialized sets of points consistent across process boundaries */
  for(int r=0; r<p4est->mpisize; ++r)
  {
    for(unsigned int m=0; m<buff_shared_projected_points_recv[r].size(); ++m)
    {
#ifdef P4_TO_P8
      Point3 p(buff_shared_projected_points_recv[r][m].x, buff_shared_projected_points_recv[r][m].y, buff_shared_projected_points_recv[r][m].z);
#else
      Point2 p(buff_shared_projected_points_recv[r][m].x, buff_shared_projected_points_recv[r][m].y);
#endif

      bool already_added = false;
      for(unsigned int k=0; k<projected_points.size(); ++k) // [Raphael]: this is terrible, maybe we should brainstorm a better approach?
      {
        if((p-projected_points[k]).norm_L2() < diag_min/close_distance_factor)
        {
          already_added = true;
          break;
        }
      }

      if(!already_added)
      {
        projected_points.push_back(p);
#ifdef P4_TO_P8
        Point3 dp(buff_shared_projected_points_recv[r][m].dx, buff_shared_projected_points_recv[r][m].dy, buff_shared_projected_points_recv[r][m].dz);
#else
        Point2 dp(buff_shared_projected_points_recv[r][m].dx, buff_shared_projected_points_recv[r][m].dy);
#endif
        projected_points_grad.push_back(dp);
      }
    }
    buff_shared_projected_points_recv[r].clear();
  }
  buff_shared_projected_points_recv.clear();

  /* add the projected local (inner) points to the list of projected points. The appropriate grid nodes have been
   * added to marked_nodes earlier */
  for(size_t i=0; i<marked_nodes.size(); ++i)
  {
    p4est_locidx_t n = marked_nodes[i];

    double xn = node_x_fr_n(n, p4est, nodes);
    double yn = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double zn = node_z_fr_n(n, p4est, nodes);
    Point3 dp((*ngbd_n).get_neighbors(n).dx_central(phi_read_only_p), (*ngbd_n).get_neighbors(n).dy_central(phi_read_only_p), (*ngbd_n).get_neighbors(n).dz_central(phi_read_only_p));
#else
    Point2 dp((*ngbd_n).get_neighbors(n).dx_central(phi_read_only_p), (*ngbd_n).get_neighbors(n).dy_central(phi_read_only_p));
#endif

    double d = phi_read_only_p[n];
    P4EST_ASSERT(dp.norm_L2() > EPS);
    dp /= dp.norm_L2();

#ifdef P4_TO_P8
    Point3 p_proj(xn-d*dp.x, yn-d*dp.y, zn-d*dp.z);
#else
    Point2 p_proj(xn-d*dp.x, yn-d*dp.y);
#endif

    bool already_added = false;
    for(unsigned int m=0; m<projected_points.size(); ++m) // [Raphael]: this is terrible, maybe we should brainstorm a better approach?
    {
      if((p_proj-projected_points[m]).norm_L2() < diag_min/close_distance_factor)
      {
        already_added = true;
        break;
      }
    }

    if(!already_added)
    {
      projected_points.push_back(p_proj);
      projected_points_grad.push_back(dp);
    }
  }
  ierr = VecRestoreArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr);
  // clear the buffers of "shared" projected points, no longer needed
  /* --- END OF INITIALIZATION OF THE PROCEDURE ---*/

  /* --- 1. CREATION OF VORONOI SEEDS ACROSS THE INTERFACE ---*/
  double band = diag_min/close_distance_factor; // distance from the created voronoi seeds to the interface
  for(unsigned int n=0; n<projected_points.size(); ++n)
  {
#ifdef P4_TO_P8
    Point3 p_proj = projected_points[n];
    Point3 dp = projected_points_grad[n];
#else
    Point2 p_proj = projected_points[n];
    Point2 dp = projected_points_grad[n];
#endif
    // add the two mirror points if they belong to the process' subregion of the domain
    for (short kk = 0; kk < 2; ++kk) {
      // calculate the voronoi seed
      double xyz [] =
      {
        std::min(xyz_max[0], std::max(xyz_min[0], p_proj.x + ((double) (1-2*kk))*band*dp.x)),
        std::min(xyz_max[1], std::max(xyz_min[1], p_proj.y + ((double) (1-2*kk))*band*dp.y))
    #ifdef P4_TO_P8
        , std::min(xyz_max[2], std::max(xyz_min[2], p_proj.z + ((double) (1-2*kk))*band*dp.z))
    #endif
      };

      // get the proc owner of that voronoi seed
      p4est_quadrant_t quad;
      std::vector<p4est_quadrant_t> remote_matches;
      int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);
      // if we are the owner, add it to our list of seeds
      if(rank_found==p4est->mpirank)
      {
        p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
        p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
        p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

        double qx = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
        double qy = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
  #ifdef P4_TO_P8
        double qz = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
  #endif

        /* add the newly created Voronoi seed to the appropriate entry in grid2voro, i.e. the entry corresponding
         * to the closest vertex of the quad owning the Voronoi seed. */
        p4est_locidx_t node = -1;
  #ifdef P4_TO_P8
        if     (xyz[0]<=qx && xyz[1]<=qy && xyz[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
        else if(xyz[0]<=qx && xyz[1]<=qy && xyz[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmp];
        else if(xyz[0]<=qx && xyz[1]> qy && xyz[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
        else if(xyz[0]<=qx && xyz[1]> qy && xyz[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpp];
        else if(xyz[0]> qx && xyz[1]<=qy && xyz[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
        else if(xyz[0]> qx && xyz[1]<=qy && xyz[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmp];
        else if(xyz[0]> qx && xyz[1]> qy && xyz[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];
        else if(xyz[0]> qx && xyz[1]> qy && xyz[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppp];
  #else
        if     (xyz[0]<=qx && xyz[1]<=qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
        else if(xyz[0]<=qx && xyz[1]> qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
        else if(xyz[0]> qx && xyz[1]<=qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
        else if(xyz[0]> qx && xyz[1]> qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];
  #endif

        grid2voro[node].push_back(voro_seeds.size());
  #ifdef P4_TO_P8
        Point3 p(xyz[0], xyz[1], xyz[2]);
  #else
        Point2 p(xyz[0], xyz[1]);
  #endif
        voro_seeds.push_back(p);
      }
    }
  }
  projected_points.clear();
  projected_points_grad.clear();
  /* --- END OF CREATION OF VORONOI SEEDS ACROSS THE INTERFACE ---*/

  /* --- 2. PARALLEL DISTRIBUTION AND GHOST LAYERING OF VORONOI SEEDS ---*/
  /* prepare the buffer to send shared local voro points */
  std::vector< std::vector<voro_seed_comm_t> > buff_send_points(p4est->mpisize);
  std::vector<bool> is_a_neighbor(p4est->mpisize, false);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    /* if the node is shared, add the relevant Voronoi seed(s) associated with
     * it to the corresponding buffer to send */
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    size_t num_sharers = (size_t) node->pad8;
    if(num_sharers>0)
    {
      sc_recycle_array_t *rec = (sc_recycle_array_t*)sc_array_index(&nodes->shared_indeps, num_sharers - 1);
      int *sharers;
      size_t sharers_index;
      if(nodes->shared_offsets == NULL)
      {
        P4EST_ASSERT(node->pad16 >= 0);
        sharers_index = (size_t) node->pad16;
      }
      else
      {
        P4EST_ASSERT(node->pad16 == -1);
        sharers_index = (size_t) nodes->shared_offsets[n];
      }
      sharers = (int*) sc_array_index(&rec->a, sharers_index);

      for(size_t s=0; s<num_sharers; ++s)
      {
        is_a_neighbor[sharers[s]] = true;

        for(unsigned int m=0; m<grid2voro[n].size(); ++m)
        {
          voro_seed_comm_t v;
          v.local_num = grid2voro[n][m];
          v.x = voro_seeds[grid2voro[n][m]].x;
          v.y = voro_seeds[grid2voro[n][m]].y;
#ifdef P4_TO_P8
          v.z = voro_seeds[grid2voro[n][m]].z;
#endif
          buff_send_points[sharers[s]].push_back(v);
        }
      }
    }
  }
  /* send the data to remote processes */
  std::vector<MPI_Request> req_send_points;
  for(int r=0; r<p4est->mpisize; ++r)
  {
    if(is_a_neighbor[r]==true)
    {
      MPI_Request req;
      mpiret = MPI_Isend((void*)&buff_send_points[r][0], buff_send_points[r].size()*sizeof(voro_seed_comm_t), MPI_BYTE, r, 2, p4est->mpicomm, &req); SC_CHECK_MPI(mpiret);
      req_send_points.push_back(req);
    }
  }
  /* get local number of voronoi points for every processor */
  num_local_voro = voro_seeds.size();
  voro_global_offset[p4est->mpirank] = num_local_voro;
  mpiret = MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &voro_global_offset[0], 1, MPI_INT, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  for(int r=1; r<p4est->mpisize; ++r)
    voro_global_offset[r] += voro_global_offset[r-1];
  voro_global_offset.insert(voro_global_offset.begin(), 0);

  nb_rcv = 0;
  for(int r=0; r<p4est->mpisize; ++r)
    if(is_a_neighbor[r]) nb_rcv++;
  /* now receive the data from neighbor process(es) */
  while(nb_rcv>0)
  {
    MPI_Status status;
    mpiret = MPI_Probe(MPI_ANY_SOURCE, 2, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);

    int nb_points;
    mpiret = MPI_Get_count(&status, MPI_BYTE, &nb_points); SC_CHECK_MPI(mpiret);
    nb_points /= sizeof(voro_seed_comm_t);

    std::vector<voro_seed_comm_t> buff_recv_points(nb_points);

    int sender_rank = status.MPI_SOURCE;
    P4EST_ASSERT(is_a_neighbor[sender_rank]);
    mpiret = MPI_Recv(&buff_recv_points[0], nb_points*sizeof(voro_seed_comm_t), MPI_BYTE, sender_rank, status.MPI_TAG, p4est->mpicomm, &status); SC_CHECK_MPI(mpiret);

    /* now associate the received voronoi points to the corresponding local/ghost nodes */
    for(int n=0; n<nb_points; ++n)
    {
      double xyz[] =
      {
        buff_recv_points[n].x,
        buff_recv_points[n].y
  #ifdef P4_TO_P8
        , buff_recv_points[n].z
  #endif
      };

      p4est_quadrant_t quad;
      std::vector<p4est_quadrant_t> remote_matches;
      int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);

      if(rank_found!=-1)
      {
        p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
        p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
        p4est_locidx_t quad_idx;
        P4EST_ASSERT(rank_found != p4est->mpirank); // this can't happen, in principle
        if(rank_found==p4est->mpirank) quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset; // this can't happen, in principle
        else                           quad_idx = quad.p.piggy3.local_num + p4est->local_num_quadrants;

        double qx = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
        double qy = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
        double qz = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
#endif

        /* add the ghost Voronoi seed to the appropriate entry in grid2voro, i.e. the entry
         * corresponding to the closest vertex of the quad owning the Voronoi seed. */
        p4est_locidx_t node = -1;
#ifdef P4_TO_P8
        if     (xyz[0]<=qx && xyz[1]<=qy && xyz[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
        else if(xyz[0]<=qx && xyz[1]<=qy && xyz[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmp];
        else if(xyz[0]<=qx && xyz[1]> qy && xyz[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
        else if(xyz[0]<=qx && xyz[1]> qy && xyz[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpp];
        else if(xyz[0]> qx && xyz[1]<=qy && xyz[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
        else if(xyz[0]> qx && xyz[1]<=qy && xyz[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmp];
        else if(xyz[0]> qx && xyz[1]> qy && xyz[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];
        else if(xyz[0]> qx && xyz[1]> qy && xyz[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppp];
#else
        if     (xyz[0]<=qx && xyz[1]<=qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
        else if(xyz[0]<=qx && xyz[1]> qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
        else if(xyz[0]> qx && xyz[1]<=qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
        else if(xyz[0]> qx && xyz[1]> qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];
#endif

        grid2voro[node].push_back(voro_seeds.size());
#ifdef P4_TO_P8
        Point3 p(xyz[0], xyz[1], xyz[2]);
#else
        Point2 p(xyz[0], xyz[1]);
#endif
        voro_seeds.push_back(p);

        voro_ghost_local_num.push_back(buff_recv_points[n].local_num);
        voro_ghost_rank.push_back(sender_rank);
      }
    }

    nb_rcv--;
  }
  /* --- END OF PARALLEL DISTRIBUTION AND GHOST LAYERING OF VORONOI SEEDS ---*/

  /* --- 3. FINALIZATION ---*/
  // finalize all non-blocking communications (the corresponding data is dynamically deleted at termination)
  mpiret = MPI_Waitall(req_shared_projected_points.size(), &req_shared_projected_points[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);
  mpiret = MPI_Waitall(req_send_points.size(), &req_send_points[0], MPI_STATUSES_IGNORE); SC_CHECK_MPI(mpiret);

  // create the RHS vector for the Voronoi jump solver, parallel layout based on the parallel layout of Voronoi seeds
  ierr = VecCreateGhostVoronoiRhs(); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_compute_voronoi_points, 0, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_poisson_jump_nodes_voronoi_t::compute_voronoi_cell(unsigned int seed_idx, Voronoi3D &voro) const
#else
void my_p4est_poisson_jump_nodes_voronoi_t::compute_voronoi_cell(unsigned int seed_idx, Voronoi2D &voro) const
#endif
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);

#ifdef P4_TO_P8
  Point3 center_seed;
#else
  Point2 center_seed;
#endif
  center_seed = voro_seeds[seed_idx];

  double xyz [] =
  {
    center_seed.x,
    center_seed.y
  #ifdef P4_TO_P8
    , center_seed.z
  #endif
  };
  /* find the quadrant to which this point belongs */
  p4est_quadrant_t owner_quad;
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, owner_quad, remote_matches);

  /* check if the point is exactly a node */
  p4est_topidx_t tree_idx = owner_quad.p.piggy3.which_tree;
  p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
  p4est_locidx_t owner_quad_idx;
  P4EST_ASSERT(rank_found != -1);
  if(rank_found==p4est->mpirank) owner_quad_idx = owner_quad.p.piggy3.local_num + tree->quadrants_offset;
  else                           owner_quad_idx = owner_quad.p.piggy3.local_num + p4est->local_num_quadrants;

  double qh[P4EST_DIM];
  dxyz_quad(p4est, &owner_quad, qh);

  double qx = quad_x_fr_q(owner_quad_idx, tree_idx, p4est, ghost);
  double qy = quad_y_fr_q(owner_quad_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
  double qz = quad_z_fr_q(owner_quad_idx, tree_idx, p4est, ghost);
#endif

#ifdef P4_TO_P8
  voro.set_center_point(seed_idx, center_seed);
#else
  voro.set_center_point(center_seed);
#endif

  // [Raphael]: tried to change the following to a std::set<p4est_locidx_t> (to avoid
  // redundant quad indices) but it does not behave well (attempts to access out-of-range
  // memory, apparently). Done another way here
  std::vector<p4est_locidx_t> indices_of_ngbd_quads;
  std::vector<p4est_quadrant_t> nb_quads;

  /* if exactly on a grid node */
  if( (fabs(xyz[0]-(qx-qh[0]/2))<(xyz_max[0] - xyz_min[0])*EPS || fabs(xyz[0]-(qx+qh[0]/2))<(xyz_max[0] - xyz_min[0])*EPS)
      && (fabs(xyz[1]-(qy-qh[1]/2))<(xyz_max[1] - xyz_min[1])*EPS || fabs(xyz[1]-(qy+qh[1]/2))<(xyz_max[1] - xyz_min[1])*EPS)
    #ifdef P4_TO_P8
      && (fabs(xyz[2]-(qz-qh[2]/2))<(xyz_max[2] - xyz_min[2])*EPS || fabs(xyz[2]-(qz+qh[2]/2))<(xyz_max[2] - xyz_min[2])*EPS)
    #endif
      )
  {
    // find that grid node
#ifdef P4_TO_P8
    int dir = (fabs(xyz[0]-(qx-qh[0]/2))<(xyz_max[0] - xyz_min[0])*EPS ?
          (fabs(xyz[1]-(qy-qh[1]/2))<(xyz_max[1] - xyz_min[1])*EPS ?
            (fabs(xyz[2]-(qz-qh[2]/2))<(xyz_max[2] - xyz_min[2])*EPS ? dir::v_mmm : dir::v_mmp)
        : (fabs(xyz[2]-(qz-qh[2]/2))<(xyz_max[2] - xyz_min[2])*EPS ? dir::v_mpm : dir::v_mpp) )
        : (fabs(xyz[1]-(qy-qh[1]/2))<(xyz_max[1] - xyz_min[1])*EPS ?
        (fabs(xyz[2]-(qz-qh[2]/2))<(xyz_max[2] - xyz_min[2])*EPS ? dir::v_pmm : dir::v_pmp)
        : (fabs(xyz[2]-(qz-qh[2]/2))<(xyz_max[2] - xyz_min[2])*EPS ? dir::v_ppm : dir::v_ppp) ) );
#else
    int dir = (fabs(xyz[0]-(qx-qh[0]/2))<(xyz_max[0] - xyz_min[0])*EPS ?
          (fabs(xyz[1]-(qy-qh[1]/2))<(xyz_max[1] - xyz_min[1])*EPS ? dir::v_mmm : dir::v_mpm)
        : (fabs(xyz[1]-(qy-qh[1]/2))<(xyz_max[1] - xyz_min[1])*EPS ? dir::v_pmm : dir::v_ppm) );
#endif
    p4est_locidx_t node = nodes->local_nodes[P4EST_CHILDREN*owner_quad_idx + dir];

    // find all the neighbor quadrants: direct neighbors + the face neighbors of
    // the direct neighbors (+ the edge neighbors of the direct neighbors in 3D)
    p4est_locidx_t direct_nb_quad_idx;
    for(char i=-1; i<=1; i+=2)
    {
      for(char j=-1; j<=1; j+=2)
      {
#ifdef P4_TO_P8
        for(char k=-1; k<=1; k+=2)
        {
          ngbd_n->find_neighbor_cell_of_node(node,  i,  j, k, direct_nb_quad_idx, tree_idx);
#else
          ngbd_n->find_neighbor_cell_of_node(node,  i,  j, direct_nb_quad_idx, tree_idx);
#endif
          if(direct_nb_quad_idx>=0) // invalid quadrant if not >=0
          {
            // check first if the direct neighbor is already in the nb_quads list
            bool add_it = true;
            for(unsigned int nn=0; nn<nb_quads.size(); ++nn)
              if(nb_quads[nn].p.piggy3.local_num==direct_nb_quad_idx)
              {
                add_it = false;
                break;
              }
            if(add_it) // add it if not in the list yet
            {
              p4est_quadrant_t tmp_quad;
              if (direct_nb_quad_idx < p4est->local_num_quadrants) {// local quadrant
                p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
                tmp_quad = *(const p4est_quadrant_t*)sc_array_index(&tree->quadrants, direct_nb_quad_idx - tree->quadrants_offset);
              } else {
                tmp_quad = *(const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, direct_nb_quad_idx - p4est->local_num_quadrants);
              }
              tmp_quad.p.piggy3.local_num = direct_nb_quad_idx;
              tmp_quad.p.piggy3.which_tree = tree_idx;
              nb_quads.push_back(tmp_quad);
            }
            // add the face neighbors of the direct neighbors, too
            // and the edge neighbors of the direct neighbors, as well in 3D
            // [Raphael]: the find_neighbor_cells_of_cell routine adds quadrant(s) to its first
            // input argument only if it is not already in so use nb_quads right away
#ifdef P4_TO_P8
            ngbd_c->find_neighbor_cells_of_cell(nb_quads, direct_nb_quad_idx, tree_idx, i, 0, 0);
            ngbd_c->find_neighbor_cells_of_cell(nb_quads, direct_nb_quad_idx, tree_idx, 0, j, 0);
            ngbd_c->find_neighbor_cells_of_cell(nb_quads, direct_nb_quad_idx, tree_idx, 0, 0, k);
            ngbd_c->find_neighbor_cells_of_cell(nb_quads, direct_nb_quad_idx, tree_idx, i, j, 0);
            ngbd_c->find_neighbor_cells_of_cell(nb_quads, direct_nb_quad_idx, tree_idx, i, 0, k);
            ngbd_c->find_neighbor_cells_of_cell(nb_quads, direct_nb_quad_idx, tree_idx, 0, j, k);
#else
            ngbd_c->find_neighbor_cells_of_cell(nb_quads, direct_nb_quad_idx, tree_idx, i, 0);
            ngbd_c->find_neighbor_cells_of_cell(nb_quads, direct_nb_quad_idx, tree_idx, 0, j);
#endif
          }
#ifdef P4_TO_P8
        }
#endif
      }
    }
    // [Raphael]: done here instead of within the nested loops to avoid redundancy in nb_quads
    for(unsigned int m=0; m<nb_quads.size(); ++m)
      indices_of_ngbd_quads.push_back(nb_quads[m].p.piggy3.local_num);
    nb_quads.clear();
  }
  /* the voronoi point is not a grid node */
  else
  {
    // add the owner quad first
    nb_quads.push_back(owner_quad);

    // then add the face neighbors of the owner quadrant
    // [Raphael] : shouldn't we add its edge neigbors as well, in 3D?
#ifdef P4_TO_P8
    // [Raphael]: the find_neighbor_cells_of_cell routine adds quadrant(s) to its first
    // input argument only if it is not already in so use nb_quads right away
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx, -1,  0,  0);
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx,  1,  0,  0);
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx,  0,  1,  0);
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx,  0, -1,  0);
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx,  0,  0, -1);
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx,  0,  0,  1);
#else
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx, -1,  0);
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx,  1,  0);
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx,  0,  1);
    ngbd_c->find_neighbor_cells_of_cell(nb_quads, owner_quad_idx, tree_idx,  0, -1);
#endif

    unsigned int current_size_of_nb_quads = nb_quads.size();

    for(unsigned int k=1; k<current_size_of_nb_quads; ++k) // skip the first one (i.e. the owner) since its face neighbors have already been added
    {
#ifdef P4_TO_P8
      // [Raphael]: the find_neighbor_cells_of_cell routine adds quadrant(s) to its first
      // input argument only if it is not already in so use nb_quads right away
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree, -1,  0,  0);
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree,  1,  0,  0);
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree,  0, -1,  0);
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree,  0,  1,  0);
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree,  0,  0, -1);
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree,  0,  0,  1);
#else
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree, -1,  0);
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree,  1,  0);
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree,  0, -1);
      ngbd_c->find_neighbor_cells_of_cell(nb_quads, nb_quads[k].p.piggy3.local_num, nb_quads[k].p.piggy3.which_tree,  0,  1);
#endif
    }

    for(unsigned int m=0; m<nb_quads.size(); ++m)
      indices_of_ngbd_quads.push_back(nb_quads[m].p.piggy3.local_num);

    // add the corner neighbors of the owner quadrant ([Raphael]: isn't that useless in 2D since
    // those are face neighbors of the face neighbors of the owner quad and thus already put in?)
    p4est_locidx_t n_idx;
    p4est_locidx_t q_idx;
    p4est_topidx_t t_idx;
    n_idx = nodes->local_nodes[owner_quad_idx*P4EST_CHILDREN + dir::v_mmm];
#ifdef P4_TO_P8
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1, -1, -1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
#else
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1, -1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
#endif
    n_idx = nodes->local_nodes[owner_quad_idx*P4EST_CHILDREN + dir::v_mpm];
#ifdef P4_TO_P8
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1,  1, -1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
#else
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1,  1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
#endif
    n_idx = nodes->local_nodes[owner_quad_idx*P4EST_CHILDREN + dir::v_pmm];
#ifdef P4_TO_P8
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1, -1, -1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
#else
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1, -1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
#endif
    n_idx = nodes->local_nodes[owner_quad_idx*P4EST_CHILDREN + dir::v_ppm];
#ifdef P4_TO_P8
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1,  1, -1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
#else
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1,  1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
#endif

#ifdef P4_TO_P8
    n_idx = nodes->local_nodes[owner_quad_idx*P4EST_CHILDREN + dir::v_mmp];
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1, -1,  1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
    n_idx = nodes->local_nodes[owner_quad_idx*P4EST_CHILDREN + dir::v_mpp];
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1,  1,  1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
    n_idx = nodes->local_nodes[owner_quad_idx*P4EST_CHILDREN + dir::v_pmp];
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1, -1,  1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
    n_idx = nodes->local_nodes[owner_quad_idx*P4EST_CHILDREN + dir::v_ppp];
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1,  1,  1, q_idx, t_idx); push_quad_idx_to_list(q_idx, indices_of_ngbd_quads);
#endif
  }

  const bool periodic[] = {is_periodic(p4est, 0), is_periodic(p4est, 1)
                         #ifdef P4_TO_P8
                           , is_periodic(p4est, 2)
                         #endif
                          };

  /* now create the list of nodes */
  for(unsigned int k = 0; k < indices_of_ngbd_quads.size(); ++k)
  {
    for(int dir=0; dir<P4EST_CHILDREN; ++dir)
    {
      p4est_locidx_t n_idx = nodes->local_nodes[P4EST_CHILDREN*indices_of_ngbd_quads[k] + dir];
      for(unsigned int m=0; m<grid2voro[n_idx].size(); ++m)
      {
        if(grid2voro[n_idx][m] != seed_idx)
        {
          // the routine push from classes Voronoi2D/3D adds the point only if its not already added
          // --> avoid duplication of points
#ifdef P4_TO_P8
          Point3 pm = voro_seeds[grid2voro[n_idx][m]];
          voro.push(grid2voro[n_idx][m], pm.x, pm.y, pm.z);
#else
          Point2 pm = voro_seeds[grid2voro[n_idx][m]];
          voro.push(grid2voro[n_idx][m], pm.x, pm.y, periodic, xyz_min, xyz_max);
#endif
        }
      }
    }
  }

  /* add the walls (hard-coded in 2D, handled by Voronoi3D in the periodicity conditions otherwise) */
#ifndef P4_TO_P8
  if(is_quad_xmWall(p4est, owner_quad.p.piggy3.which_tree, &owner_quad)) voro.push(WALL_m00, center_seed.x-MAX((xyz_max[0]-xyz_min[0])*EPS, 2*(center_seed.x-xyz_min[0])), center_seed.y , periodic, xyz_min, xyz_max);
  if(is_quad_xpWall(p4est, owner_quad.p.piggy3.which_tree, &owner_quad)) voro.push(WALL_p00, center_seed.x+MAX((xyz_max[0]-xyz_min[0])*EPS, 2*(xyz_max[0]-center_seed.x)), center_seed.y , periodic, xyz_min, xyz_max);
  if(is_quad_ymWall(p4est, owner_quad.p.piggy3.which_tree, &owner_quad)) voro.push(WALL_0m0, center_seed.x, center_seed.y-MAX((xyz_max[1]-xyz_min[1])*EPS, 2*(center_seed.y-xyz_min[1])) , periodic, xyz_min, xyz_max);
  if(is_quad_ypWall(p4est, owner_quad.p.piggy3.which_tree, &owner_quad)) voro.push(WALL_0p0, center_seed.x, center_seed.y+MAX((xyz_max[1]-xyz_min[1])*EPS, 2*(xyz_max[1]-center_seed.y)) , periodic, xyz_min, xyz_max);
#endif

  /* finally, construct the partition */
#ifdef P4_TO_P8
  voro.construct_partition(xyz_min, xyz_max, periodic);
#else
    voro.construct_partition();
#endif

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_jump_nodes_voronoi_t::setup_linear_system()
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  std::vector< std::vector<mat_entry_t> > matrix_entries(num_local_voro);
  std::vector<PetscInt> d_nnz(num_local_voro, 1), o_nnz(num_local_voro, 0);

  for(unsigned int n=0; n<num_local_voro; ++n)
  {
    PetscInt global_n_idx = n+voro_global_offset[p4est->mpirank];

#ifdef P4_TO_P8
    Point3 pc = voro_seeds[n];
#else
    Point2 pc = voro_seeds[n];
#endif
    if( ( ((!is_periodic(p4est, 0)) && (ABS(pc.x-xyz_min[0])<(xyz_max[0]-xyz_min[0])*EPS || ABS(pc.x-xyz_max[0])<(xyz_max[0]-xyz_min[0])*EPS))
         || ( (!is_periodic(p4est, 1)) && (ABS(pc.y-xyz_min[1])<(xyz_max[1]-xyz_min[1])*EPS || ABS(pc.y-xyz_max[1])<(xyz_max[1]-xyz_min[1])*EPS))
     #ifdef P4_TO_P8
         || ( (!is_periodic(p4est, 2)) && (ABS(pc.z-xyz_min[2])<(xyz_max[2]-xyz_min[2])*EPS || ABS(pc.z-xyz_max[2])<(xyz_max[2]-xyz_min[2])*EPS))
         ) && bc->wallType(pc.x,pc.y, pc.z)==DIRICHLET)
#else
         ) && bc->wallType(pc.x,pc.y)==DIRICHLET)
#endif
    {
      matrix_has_nullspace = false;
      mat_entry_t ent; ent.n = global_n_idx; ent.val = 1;
      matrix_entries[n].push_back(ent);
#ifdef P4_TO_P8
      rhs_p[n] = bc->wallValue(pc.x, pc.y, pc.z);
#else
      rhs_p[n] = bc->wallValue(pc.x, pc.y);
#endif

      continue;
    }

#ifdef P4_TO_P8
    Voronoi3D voro;
#else
    Voronoi2D voro;
#endif
    compute_voronoi_cell(n, voro);

#ifdef P4_TO_P8
    const std::vector<ngbd3Dseed> *neighbor_seeds;
#else
    const std::vector<Point2> *partition;
    const std::vector<ngbd2Dseed> *neighbor_seeds;
    voro.get_partition(partition);
#endif
    voro.get_neighbor_seeds(neighbor_seeds);

#ifdef P4_TO_P8
    double phi_n = interp_phi(pc.x, pc.y, pc.z);
#else
    double phi_n = interp_phi(pc.x, pc.y);
#endif
    double mu_n, add_n;

    if(phi_n<0)
    {
#ifdef P4_TO_P8
      rhs_p[n] = this->rhs_m(pc.x, pc.y, pc.z);
      mu_n     = (*mu_m)(pc.x, pc.y, pc.z);
      add_n    = (*add_m)(pc.x, pc.y, pc.z);
#else
      rhs_p[n] = this->rhs_m(pc.x, pc.y);
      mu_n     = (*mu_m)(pc.x, pc.y);
      add_n    = (*add_m)(pc.x, pc.y);
#endif
    }
    else
    {
#ifdef P4_TO_P8
      rhs_p[n] = this->rhs_p(pc.x, pc.y, pc.z);
      mu_n     = (*mu_p)(pc.x, pc.y, pc.z);
      add_n    = (*add_p)(pc.x, pc.y, pc.z);
#else
      rhs_p[n] = this->rhs_p(pc.x, pc.y);
      mu_n     = (*mu_p)(pc.x, pc.y);
      add_n    = (*add_p)(pc.x, pc.y);
#endif
    }

#ifndef P4_TO_P8
    voro.compute_volume();
#endif
    double volume = voro.get_volume();

    rhs_p[n] *= volume;
    if(add_n>EPS) matrix_has_nullspace = false;

    mat_entry_t ent; ent.n = global_n_idx; ent.val = volume*add_n;
    matrix_entries[n].push_back(ent);

    for(unsigned int l=0; l<neighbor_seeds->size(); ++l)
    {
#ifdef P4_TO_P8
      double s = (*neighbor_seeds)[l].s;
#else
      int k = (l+partition->size()-1) % partition->size();
      double s = ((*partition)[k]-(*partition)[l]).norm_L2();
#endif

      if((*neighbor_seeds)[l].n>=0)
      {
        /* regular point */
#ifdef P4_TO_P8
        Point3 pl = (*neighbor_seeds)[l].p;
        double phi_l = interp_phi(pl.x, pl.y, pl.z);
#else
        Point2 pl = (*neighbor_seeds)[l].p;
        double phi_l = interp_phi(pl.x, pl.y);
#endif
        double d = (pc - pl).norm_L2();
        double mu_l;

#ifdef P4_TO_P8
        if(phi_l<0) mu_l = (*mu_m)(pl.x, pl.y, pl.z);
        else        mu_l = (*mu_p)(pl.x, pl.y, pl.z);
#else
        if(phi_l<0) mu_l = (*mu_m)(pl.x, pl.y);
        else        mu_l = (*mu_p)(pl.x, pl.y);
#endif

        double mu_harmonic = 2*mu_n*mu_l/(mu_n + mu_l);

        PetscInt global_l_idx;
        if((unsigned int)(*neighbor_seeds)[l].n<num_local_voro)
        {
          global_l_idx = (*neighbor_seeds)[l].n + voro_global_offset[p4est->mpirank];
          d_nnz[n]++;
        }
        else
        {
          global_l_idx = voro_ghost_local_num[(*neighbor_seeds)[l].n-num_local_voro] + voro_global_offset[voro_ghost_rank[(*neighbor_seeds)[l].n-num_local_voro]];
          o_nnz[n]++;
        }

        mat_entry_t ent; ent.n = global_l_idx; ent.val = -s*mu_harmonic/d;
        matrix_entries[n][0].val += s*mu_harmonic/d;
        matrix_entries[n].push_back(ent);

        if(phi_n*phi_l<0)
        {
#ifdef P4_TO_P8
          Point3 p_ln = (pc+pl)/2;
#else
          Point2 p_ln = (pc+pl)/2;
#endif

#ifdef P4_TO_P8
          rhs_p[n] += s*mu_harmonic/d * SIGN(phi_n) * (*u_jump)(p_ln.x, p_ln.y, p_ln.z);
          rhs_p[n] -= mu_harmonic/mu_l * s/2 * (*mu_grad_u_jump)(p_ln.x, p_ln.y, p_ln.z);
#else
          rhs_p[n] += s*mu_harmonic/d * SIGN(phi_n) * (*u_jump)(p_ln.x, p_ln.y);
          rhs_p[n] -= mu_harmonic/mu_l * s/2 * (*mu_grad_u_jump)(p_ln.x, p_ln.y);
#endif
        }
      }
      else /* wall with neumann */
      {
        double x_tmp = pc.x;
        double y_tmp = pc.y;

        /* perturb the corners to differentiate between the edges of the domain ... otherwise 1st order only at the corners */
#ifdef P4_TO_P8
        double z_tmp = pc.z;
        if(fabs(pc.x-xyz_min[0])<(xyz_max[0]-xyz_min[0])*EPS && ( (*neighbor_seeds)[l].n==WALL_0m0  || (*neighbor_seeds)[l].n==WALL_0p0 || (*neighbor_seeds)[l].n==WALL_00m  || (*neighbor_seeds)[l].n==WALL_00p) ) x_tmp += 2*EPS*(xyz_max[0]-xyz_min[0]);
        if(fabs(pc.x-xyz_max[0])<(xyz_max[0]-xyz_min[0])*EPS && ( (*neighbor_seeds)[l].n==WALL_0m0  || (*neighbor_seeds)[l].n==WALL_0p0 || (*neighbor_seeds)[l].n==WALL_00m  || (*neighbor_seeds)[l].n==WALL_00p) ) x_tmp -= 2*EPS*(xyz_max[0]-xyz_min[0]);
        if(fabs(pc.y-xyz_min[1])<(xyz_max[1]-xyz_min[1])*EPS && ( (*neighbor_seeds)[l].n==WALL_m00  || (*neighbor_seeds)[l].n==WALL_p00 || (*neighbor_seeds)[l].n==WALL_00m  || (*neighbor_seeds)[l].n==WALL_00p) ) y_tmp += 2*EPS*(xyz_max[1]-xyz_min[1]);
        if(fabs(pc.y-xyz_max[1])<(xyz_max[1]-xyz_min[1])*EPS && ( (*neighbor_seeds)[l].n==WALL_m00  || (*neighbor_seeds)[l].n==WALL_p00 || (*neighbor_seeds)[l].n==WALL_00m  || (*neighbor_seeds)[l].n==WALL_00p) ) y_tmp -= 2*EPS*(xyz_max[1]-xyz_min[1]);
        if(fabs(pc.z-xyz_min[2])<(xyz_max[2]-xyz_min[2])*EPS && ( (*neighbor_seeds)[l].n==WALL_m00  || (*neighbor_seeds)[l].n==WALL_p00 || (*neighbor_seeds)[l].n==WALL_0m0  || (*neighbor_seeds)[l].n==WALL_0p0) ) z_tmp += 2*EPS*(xyz_max[2]-xyz_min[2]);
        if(fabs(pc.z-xyz_max[2])<(xyz_max[2]-xyz_min[2])*EPS && ( (*neighbor_seeds)[l].n==WALL_m00  || (*neighbor_seeds)[l].n==WALL_p00 || (*neighbor_seeds)[l].n==WALL_0m0  || (*neighbor_seeds)[l].n==WALL_0p0) ) z_tmp -= 2*EPS*(xyz_max[2]-xyz_min[2]);
        rhs_p[n] += s*mu_n * bc->wallValue(x_tmp, y_tmp, z_tmp);
#else
        if(fabs(pc.x-xyz_min[0])<(xyz_max[0]-xyz_min[0])*EPS && ((*neighbor_seeds)[l].n==WALL_0m0  || (*neighbor_seeds)[l].n==WALL_0p0)) x_tmp += 2*EPS*(xyz_max[0]-xyz_min[0]);
        if(fabs(pc.x-xyz_max[0])<(xyz_max[0]-xyz_min[0])*EPS && ((*neighbor_seeds)[l].n==WALL_0m0  || (*neighbor_seeds)[l].n==WALL_0p0)) x_tmp -= 2*EPS*(xyz_max[0]-xyz_min[0]);
        if(fabs(pc.y-xyz_min[1])<(xyz_max[1]-xyz_min[1])*EPS && ((*neighbor_seeds)[l].n==WALL_m00  || (*neighbor_seeds)[l].n==WALL_p00)) y_tmp += 2*EPS*(xyz_max[1]-xyz_min[1]);
        if(fabs(pc.y-xyz_max[1])<(xyz_max[1]-xyz_min[1])*EPS && ((*neighbor_seeds)[l].n==WALL_m00  || (*neighbor_seeds)[l].n==WALL_p00)) y_tmp -= 2*EPS*(xyz_max[1]-xyz_min[1]);
        rhs_p[n] += s*mu_n * bc->wallValue(x_tmp, y_tmp);
#endif
      }
    }

//    double scaling_value = (matrix_entries[n].size() > 1)? 0.0:1.0;
//    for(size_t l=1; l<matrix_entries[n].size(); ++l)
//      scaling_value += fabs(matrix_entries[n][l].val);

//    for(size_t l=0; l<matrix_entries[n].size(); ++l)
//    {
//      matrix_entries[n][l].val  /= scaling_value;
//    }
//    rhs_p[n]                    /= scaling_value;
  }

  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  PetscInt num_owned_global = voro_global_offset[p4est->mpisize];
  PetscInt num_owned_local  = (PetscInt) num_local_voro;

  if (A != NULL){
    ierr = MatDestroy(A); CHKERRXX(ierr);}

  /* set up the matrix */
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  /* allocate the matrix */
  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  /* fill the matrix with the values */
  for(unsigned int n=0; n<num_local_voro; ++n)
  {
    PetscInt global_n_idx = n+voro_global_offset[p4est->mpirank];
    for(unsigned int m=0; m<matrix_entries[n].size(); ++m)
      ierr = MatSetValue(A, global_n_idx, matrix_entries[n][m].n, matrix_entries[n][m].val, ADD_VALUES); CHKERRXX(ierr);
  }
  /* assemble the matrix */
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

  /* check for null space */
  MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace, 1, MPI_INT, MPI_LAND, p4est->mpicomm);
  if (matrix_has_nullspace)
  {
    if (A_null_space == NULL)
    {
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, PETSC_NULL, &A_null_space); CHKERRXX(ierr);
    }
    ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);


//  const std::string name = "/home/egan/workspace/projects/jump_solver/output/matrix_" +std::to_string(xyz_max[0]-xyz_min[0]);
//  const std::string name_rhs = "/home/egan/workspace/projects/jump_solver/output/rhs_" +std::to_string(xyz_max[0]-xyz_min[0]);
//  PetscViewer viewer, viewer_rhs;
//  ierr = PetscViewerASCIIOpen(p4est->mpicomm, name.c_str(), &viewer); CHKERRXX(ierr);
//  ierr = PetscViewerASCIIOpen(p4est->mpicomm, name_rhs.c_str(), &viewer_rhs); CHKERRXX(ierr);
//  ierr = MatView(A, viewer); CHKERRXX(ierr);
//  ierr = VecView(rhs, viewer_rhs); CHKERRXX(ierr);
//  ierr = PetscViewerDestroy(viewer); CHKERRXX(ierr);
//  ierr = PetscViewerDestroy(viewer_rhs); CHKERRXX(ierr);

}


void my_p4est_poisson_jump_nodes_voronoi_t::setup_negative_laplace_rhsvec()
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  for(unsigned int n=0; n<num_local_voro; ++n)
  {
#ifdef P4_TO_P8
    Point3 pc = voro_seeds[n];
#else
    Point2 pc = voro_seeds[n];
#endif
    if( (ABS(pc.x-xyz_min[0])<(xyz_max[0]-xyz_min[0])*EPS || ABS(pc.x-xyz_max[0])<(xyz_max[0]-xyz_min[0])*EPS ||
         ABS(pc.y-xyz_min[1])<(xyz_max[1]-xyz_min[1])*EPS || ABS(pc.y-xyz_max[1])<(xyz_max[1]-xyz_min[1])*EPS
     #ifdef P4_TO_P8
         || ABS(pc.z-xyz_min[2])<(xyz_max[2]-xyz_min[2])*EPS || ABS(pc.z-xyz_max[2])<(xyz_max[2]-xyz_min[2])*EPS
         ) && bc->wallType(pc.x,pc.y, pc.z)==DIRICHLET)
#else
         ) && bc->wallType(pc.x,pc.y)==DIRICHLET)
#endif
    {
#ifdef P4_TO_P8
      rhs_p[n] = bc->wallValue(pc.x, pc.y, pc.z);
#else
      rhs_p[n] = bc->wallValue(pc.x, pc.y);
#endif
      continue;
    }

#ifdef P4_TO_P8
    Voronoi3D voro;
#else
    Voronoi2D voro;
#endif
    compute_voronoi_cell(n, voro);

#ifdef P4_TO_P8
    const std::vector<ngbd3Dseed> *points;
#else
    const std::vector<Point2> *partition;
    const std::vector<ngbd2Dseed> *points;
    voro.get_partition(partition);
#endif
    voro.get_neighbor_seeds(points);

#ifdef P4_TO_P8
    double phi_n = interp_phi(pc.x, pc.y, pc.z);
#else
    double phi_n = interp_phi(pc.x, pc.y);
#endif
    double mu_n;

    if(phi_n<0)
    {
#ifdef P4_TO_P8
      rhs_p[n] = this->rhs_m(pc.x, pc.y, pc.z);
      mu_n     = (*mu_m)(pc.x, pc.y, pc.z);
#else
      rhs_p[n] = this->rhs_m(pc.x, pc.y);
      mu_n     = (*mu_m)(pc.x, pc.y);
#endif
    }
    else
    {
#ifdef P4_TO_P8
      rhs_p[n] = this->rhs_p(pc.x, pc.y, pc.z);
      mu_n     = (*mu_p)(pc.x, pc.y, pc.z);
#else
      rhs_p[n] = this->rhs_p(pc.x, pc.y);
      mu_n     = (*mu_p)(pc.x, pc.y);
#endif
    }

#ifndef P4_TO_P8
    voro.compute_volume();
#endif
    rhs_p[n] *= voro.get_volume();

    for(unsigned int l=0; l<points->size(); ++l)
    {
#ifdef P4_TO_P8
      double s = (*points)[l].s;
#else
      int k = (l+partition->size()-1) % partition->size();
      double s = ((*partition)[k]-(*partition)[l]).norm_L2();
#endif
      if((*points)[l].n>=0)
      {
#ifdef P4_TO_P8
        Point3 pl = (*points)[l].p;
        double phi_l = interp_phi(pl.x, pl.y, pl.z);
#else
        Point2 pl = (*points)[l].p;
        double phi_l = interp_phi(pl.x, pl.y);
#endif
        double d = (pc - pl).norm_L2();
        if(phi_n*phi_l<0)
        {
          double mu_l;
#ifdef P4_TO_P8
          if(phi_l<0) mu_l = (*mu_m)(pl.x, pl.y, pl.z);
          else        mu_l = (*mu_p)(pl.x, pl.y, pl.z);
          Point3 p_ln = (pc+pl)/2;
#else
          if(phi_l<0) mu_l = (*mu_m)(pl.x, pl.y);
          else        mu_l = (*mu_p)(pl.x, pl.y);
          Point2 p_ln = (pc+pl)/2;
#endif

          double mu_harmonic = 2*mu_n*mu_l/(mu_n + mu_l);

#ifdef P4_TO_P8
          rhs_p[n] += s*mu_harmonic/d * SIGN(phi_n) * (*u_jump)(p_ln.x, p_ln.y, p_ln.z);
          rhs_p[n] -= mu_harmonic/mu_l * s/2 * (*mu_grad_u_jump)(p_ln.x, p_ln.y, p_ln.z);
#else
          rhs_p[n] += s*mu_harmonic/d * SIGN(phi_n) * (*u_jump)(p_ln.x, p_ln.y);
          rhs_p[n] -= mu_harmonic/mu_l * s/2 * (*mu_grad_u_jump)(p_ln.x, p_ln.y);
#endif
        }
      }
      else /* wall with neumann */
      {
        double x_tmp = pc.x;
        double y_tmp = pc.y;

        /* perturb the corners to differentiate between the edges of the domain ... otherwise 1st order only at the corners */
#ifdef P4_TO_P8
        double z_tmp = pc.z;
        if(fabs(pc.x-xyz_min[0])<(xyz_max[0]-xyz_min[0])*EPS && ( (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp += 2*EPS*(xyz_max[0]-xyz_min[0]);
        if(fabs(pc.x-xyz_max[0])<(xyz_max[0]-xyz_min[0])*EPS && ( (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp -= 2*EPS*(xyz_max[0]-xyz_min[0]);
        if(fabs(pc.y-xyz_min[1])<(xyz_max[1]-xyz_min[1])*EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp += 2*EPS*(xyz_max[1]-xyz_min[1]);
        if(fabs(pc.y-xyz_max[1])<(xyz_max[1]-xyz_min[1])*EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp -= 2*EPS*(xyz_max[1]-xyz_min[1]);
        if(fabs(pc.z-xyz_min[2])<(xyz_max[2]-xyz_min[2])*EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp += 2*EPS*(xyz_max[2]-xyz_min[2]);
        if(fabs(pc.z-xyz_max[2])<(xyz_max[2]-xyz_min[2])*EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp -= 2*EPS*(xyz_max[2]-xyz_min[2]);
        rhs_p[n] += s*mu_n * bc->wallValue(x_tmp, y_tmp, z_tmp);
#else
        if(fabs(pc.x-xyz_min[0])<(xyz_max[0]-xyz_min[0])*EPS && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp += 2*EPS*(xyz_max[0]-xyz_min[0]);
        if(fabs(pc.x-xyz_max[0])<(xyz_max[0]-xyz_min[0])*EPS && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp -= 2*EPS*(xyz_max[0]-xyz_min[0]);
        if(fabs(pc.y-xyz_min[1])<(xyz_max[1]-xyz_min[1])*EPS && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp += 2*EPS*(xyz_max[1]-xyz_min[1]);
        if(fabs(pc.y-xyz_max[1])<(xyz_max[1]-xyz_min[1])*EPS && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp -= 2*EPS*(xyz_max[1]-xyz_min[1]);
        rhs_p[n] += s*mu_n * bc->wallValue(x_tmp, y_tmp);
#endif
      }
    }
  }

  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  if (matrix_has_nullspace)
    ierr = MatNullSpaceRemove(A_null_space, rhs, NULL); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);
}



double my_p4est_poisson_jump_nodes_voronoi_t::interpolate_solution_from_voronoi_to_tree_on_node_n(p4est_locidx_t node_idx) const
{
  PetscErrorCode ierr;

  const double *sol_voro_read_only_p;
  ierr = VecGetArrayRead(sol_voro, &sol_voro_read_only_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
  Point3 p_node(node_x_fr_n(node_idx, p4est, nodes), node_y_fr_n(node_idx, p4est, nodes), node_z_fr_n(node_idx, p4est, nodes));
#else
  Point2 p_node(node_x_fr_n(node_idx, p4est, nodes), node_y_fr_n(node_idx, p4est, nodes));
#endif

#ifdef P4_TO_P8
  Point3 p_seed;
#else
  Point2 p_seed;
#endif
  /* first check if the node is a voronoi point */
  for(unsigned int m=0; m<grid2voro[node_idx].size(); ++m)
  {
    p_seed = voro_seeds[grid2voro[node_idx][m]];
    if((p_node-p_seed).norm_L2()<sqrt(SQR(xyz_max[0]-xyz_min[0])+SQR(xyz_max[1]-xyz_min[1])
                          #ifdef P4_TO_P8
                                +SQR(xyz_max[2]-xyz_min[2])
                          #endif
                                )*EPS)
    {
      double retval = sol_voro_read_only_p[grid2voro[node_idx][m]];
      ierr = VecRestoreArrayRead(sol_voro, &sol_voro_read_only_p); CHKERRXX(ierr);
      return retval;
    }
  }

  const double *phi_read_only_p;
  ierr = VecGetArrayRead(phi, &phi_read_only_p); CHKERRXX(ierr);

  /* if not a grid point, gather all the neighbor voro points and find the
   * three closest with the same sign for phi */
  p4est_locidx_t quad_idx;
  p4est_topidx_t tree_idx;

  std::vector<p4est_locidx_t> indices_of_ngbd_quads;
  std::vector<p4est_quadrant_t> nb_quads;
  for(char i=-1; i<=1; i+=2)
  {
    for(char j=-1; j<=1; j+=2)
    {
#ifdef P4_TO_P8
      for(char k=-1; k<=1; k+=2)
      {
        ngbd_n->find_neighbor_cell_of_node(node_idx,  i,  j, k, quad_idx, tree_idx);
#else
        ngbd_n->find_neighbor_cell_of_node(node_idx,  i,  j, quad_idx, tree_idx);
#endif
        if(quad_idx>=0)
        {
          // check first if the direct neighbor is already in the nb_quads list
          bool add_it = true;
          for(unsigned int nn=0; nn<nb_quads.size(); ++nn)
            if(nb_quads[nn].p.piggy3.local_num==quad_idx)
            {
              add_it = false;
              break;
            }
          if(add_it) // add it if not in the list yet
          {
            p4est_quadrant_t tmp_quad;
            if (quad_idx < p4est->local_num_quadrants) {// local quadrant
              p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
              tmp_quad = *(const p4est_quadrant_t*)sc_array_index(&tree->quadrants, quad_idx - tree->quadrants_offset);
            } else {
              tmp_quad = *(const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx - p4est->local_num_quadrants);
            }
            tmp_quad.p.piggy3.local_num = quad_idx;
            tmp_quad.p.piggy3.which_tree = tree_idx;
            nb_quads.push_back(tmp_quad);
          }
          // add the face, edge and corner neighbors of the direct neighbors, too, in the same direction(s)
          // [Raphael]: the find_neighbor_cells_of_cell routine adds quadrant(s) to its first
          // input argument only if it is not already in so use nb_quads right away
#ifdef P4_TO_P8
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, i, 0, 0);
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, 0, j, 0);
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, 0, 0, k);
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, i, j, 0);
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, i, 0, k);
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, 0, j, k);
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, i, j, k);
#else
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, i, 0);
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, 0, j);
          ngbd_c->find_neighbor_cells_of_cell(nb_quads, quad_idx, tree_idx, i, j);
#endif
        }
#ifdef P4_TO_P8
      }
#endif
    }
  }
  // [Raphael]: done here instead of within the nested loops to avoid redundancy in nb_quads
  for(unsigned int m=0; m<nb_quads.size(); ++m)
    indices_of_ngbd_quads.push_back(nb_quads[m].p.piggy3.local_num);
  nb_quads.clear();

  /* Find (P4EST_DIM + 1) appropriate points for linear interpolation of the solution.
   * We start by sorting the points, on the same side of the interface, from the closest
   * to the farthest */
  double phi_node = phi_read_only_p[node_idx];
#ifdef P4_TO_P8
  double di[] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
  unsigned int ni[] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
#else
  double di[] = {DBL_MAX, DBL_MAX, DBL_MAX};
  unsigned int ni[] = {UINT_MAX, UINT_MAX, UINT_MAX};
#endif
  std::vector<neighbor_seed> nb_seeds;
  for(unsigned int k=0; k<indices_of_ngbd_quads.size(); ++k)
  {
    for(int dir=0; dir<P4EST_CHILDREN; ++dir)
    {
      p4est_locidx_t n_idx = nodes->local_nodes[P4EST_CHILDREN*indices_of_ngbd_quads[k] + dir];
      for(unsigned int m=0; m<grid2voro[n_idx].size(); ++m)
      {
        size_t seed_idx = grid2voro[n_idx][m];
        bool not_yet_in = true;
        for (size_t mm = 0; mm < nb_seeds.size(); ++mm) {
          if(seed_idx == nb_seeds[mm].local_seed_idx)
          {
            not_yet_in = false;
            break;
          }
        }
        if(not_yet_in)
        {
          p_seed = voro_seeds[seed_idx];
#ifdef P4_TO_P8
          double xyz[] = {p_seed.x, p_seed.y, p_seed.z};
#else
          double xyz[] = {p_seed.x, p_seed.y};
#endif
          p4est_quadrant_t quad;
          std::vector<p4est_quadrant_t> remote_matches;

          int rank = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);
          if(rank!=-1)
          {
#ifdef P4_TO_P8
            double phi_m = interp_phi(p_seed.x, p_seed.y, p_seed.z);
#else
            double phi_m = interp_phi(p_seed.x, p_seed.y);
#endif
            if(phi_m*phi_node>=0)
            {
              neighbor_seed ngbd_seed;
              ngbd_seed.local_seed_idx = seed_idx;
              ngbd_seed.distance = (p_seed-p_node).norm_L2();
              nb_seeds.push_back(ngbd_seed);
            }
          } else {
            throw std::runtime_error("Found rank = -1 in voronoi interpolaiton. This should not happen. SHIT SHIT SHIT");
          }
        }
      }
    }
  }

  std::sort(nb_seeds.begin(), nb_seeds.end()); // sort the neighbor seeds from the closest to the farthest
  /* Start with the two closest voronoi seeds */
  P4EST_ASSERT((nb_seeds.size() > 1));
  ni[0] = nb_seeds[0].local_seed_idx; di[0] = nb_seeds[0].distance;
  ni[1] = nb_seeds[1].local_seed_idx; di[1] = nb_seeds[1].distance;

#ifdef P4_TO_P8
  Point3 p0(voro_seeds[ni[0]]);
  Point3 p1(voro_seeds[ni[1]]);
#else
  Point2 p0(voro_seeds[ni[0]]);
  Point2 p1(voro_seeds[ni[1]]);
#endif

  /* find a third point forming a non-flat triangle */
  P4EST_ASSERT((nb_seeds.size() > 2));
  for(unsigned int k=2; k<nb_seeds.size(); ++k) // start from k = 2 since the two first ones are already in
  {
    p_seed = voro_seeds[nb_seeds[k].local_seed_idx];
    // no need to check if the point is local and/or on the right side of the interface, already done earlier
#ifdef P4_TO_P8
    if( (((p0-p1).normalize().cross((p_seed-p1).normalize())).norm_L2() > sin(PI/10)) && (((p0-p1).normalize().cross((p_seed-p0).normalize())).norm_L2() > sin(PI/10)) )
#else
    if( (ABS((p0-p1).normalize().cross((p_seed-p1).normalize())) > sin(PI/5)) && (ABS((p0-p1).normalize().cross((p_seed-p0).normalize())) > sin(PI/5)) )
#endif
    {
      // non-degenerate triangle found, and this is the closest point by construction
      ni[2]=nb_seeds[k].local_seed_idx; di[2]=nb_seeds[k].distance;
      break;
    }
  }

#ifdef P4_TO_P8
  Point3 p2(voro_seeds[ni[2]]);
#else
  Point2 p2(voro_seeds[ni[2]]);
#endif

#ifdef P4_TO_P8
  /* in 3D, found a fourth point to form a non-flat tetrahedron */
  P4EST_ASSERT((nb_seeds.size() > 3));
  for(unsigned int k=2; k<nb_seeds.size(); ++k) // start from k = 2 since the two first ones are already in
  {
    if(nb_seeds[k].local_seed_idx == ni[2]) // this is the point added earlier
      continue;
    p_seed = voro_seeds[nb_seeds[k].local_seed_idx];
    // no need to check if the point is local and/or on the right side of the interface, already done earlier
    Point3 n = (p1-p0).cross(p2-p0).normalize();
    double h = ABS((p_seed-p0).dot(n));
    if(h > 2.0*diag_min/close_distance_factor)
    {
      // non-degenerate tetrahedron found, and this is the closest point by construction
      ni[3]=nb_seeds[k].local_seed_idx; di[3]=nb_seeds[k].distance;
      break;
    }
  }
  Point3 p3(voro_seeds[ni[3]]);
#endif

  /* make sure we found (P4EST_DIM + 1) points */
#ifdef P4_TO_P8
  if(di[0]==DBL_MAX || di[1]==DBL_MAX || di[2]==DBL_MAX || di[3]==DBL_MAX)
#else
  if(di[0]==DBL_MAX || di[1]==DBL_MAX || di[2]==DBL_MAX)
#endif
  {
    std::cerr << "my_p4est_poisson_jump_nodes_voronoi_t->interpolate_solution_from_voronoi_to_tree: not enough points found." << std::endl;
    double retval = sol_voro_read_only_p[ni[0]];
    ierr = VecRestoreArrayRead(phi     , &phi_read_only_p     ); CHKERRXX(ierr);
    ierr = VecRestoreArrayRead(sol_voro, &sol_voro_read_only_p); CHKERRXX(ierr);
    return retval;
  }

  // print a warning if a point is duplicated
#ifdef P4_TO_P8
  if(ni[0]==ni[1] || ni[0]==ni[2] || ni[0]==ni[3] || ni[1]==ni[2] || ni[1]==ni[3] || ni[2]==ni[3])
#else
  if(ni[0]==ni[1] || ni[0]==ni[2] || ni[1]==ni[2])
#endif
    std::cerr << "my_p4est_poisson_jump_nodes_voronoi_t->interpolate_solution_from_voronoi_to_tree: point is double !" << std::endl;

  double f0 = sol_voro_read_only_p[ni[0]];
  double f1 = sol_voro_read_only_p[ni[1]];
  double f2 = sol_voro_read_only_p[ni[2]];
#ifdef P4_TO_P8
  double f3 = sol_voro_read_only_p[ni[3]];
#endif
  ierr = VecRestoreArrayRead(phi     , &phi_read_only_p     ); CHKERRXX(ierr);
  ierr = VecRestoreArrayRead(sol_voro, &sol_voro_read_only_p); CHKERRXX(ierr);

  p0 -= p_node;
  p1 -= p_node;
  p2 -= p_node;
#ifdef P4_TO_P8
  p3 -= p_node;

  double den  = ((p1-p0).cross(p2-p0)).dot(p3-p0);
  double c0   = (p1.cross(p2)).dot(p3)/den;
  double c1   =-(p0.cross(p2)).dot(p3)/den;
  double c2   = (p0.cross(p1)).dot(p3)/den;
  double c3   = 1.0-c0-c1-c2;

  return (f0*c0+f1*c1+f2*c2+f3*c3);
#else

  double den  = (p1-p0).cross(p2-p0);

  double c0 = (p1.cross(p2))/den;
  double c1 = -(p0.cross(p2))/den;
  double c2 = 1.0-c0-c1;

  return (f0*c0+f1*c1+f2*c2);
#endif
}



void my_p4est_poisson_jump_nodes_voronoi_t::interpolate_solution_from_voronoi_to_tree(Vec solution) const
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_interpolate_to_tree, phi, sol_voro, solution, 0); CHKERRXX(ierr);

  double *solution_p;
  ierr = VecGetArray(solution, &solution_p); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    solution_p[n] = interpolate_solution_from_voronoi_to_tree_on_node_n(n);
  }

  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    solution_p[n] = interpolate_solution_from_voronoi_to_tree_on_node_n(n);
  }

  ierr = VecGhostUpdateEnd  (solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = VecRestoreArray(solution, &solution_p); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_interpolate_to_tree, phi, sol_voro, solution, 0); CHKERRXX(ierr);
}



void my_p4est_poisson_jump_nodes_voronoi_t::write_stats(const char *path) const
{
  std::vector<unsigned int> voro_seeds_that_are_local_grid_nodes(p4est->mpisize, 0);
  std::vector<unsigned int> voro_seeds_that_are_ghost_grid_nodes(p4est->mpisize, 0);
  std::vector<unsigned int> independent_voro_seeds_that_are_close_to_local_grid_nodes(p4est->mpisize, 0);
  std::vector<unsigned int> independent_voro_seeds_that_are_close_to_ghost_grid_nodes(p4est->mpisize, 0);

  for(p4est_locidx_t n=0; n< nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    Point3 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes), node_z_fr_n(n, p4est, nodes));
#else
    Point2 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes));
#endif

    /* first check if the node is a voronoi point */
    for(unsigned int m=0; m<grid2voro[n].size(); ++m)
    {
      if(((pn-voro_seeds[grid2voro[n][m]]).norm_L2()< sqrt(SQR(xyz_max[0]-xyz_min[0])+SQR(xyz_max[1]-xyz_min[1])
                                                     #ifdef P4_TO_P8
                                                           +SQR(xyz_max[2]-xyz_min[2])
                                                     #endif
                                                           )*EPS) && (grid2voro[n][m] < num_local_voro))
        voro_seeds_that_are_local_grid_nodes[p4est->mpirank]++;
      else if(grid2voro[n][m] < num_local_voro)
        independent_voro_seeds_that_are_close_to_local_grid_nodes[p4est->mpirank]++;
    }
  }

  for(p4est_locidx_t n=nodes->num_owned_indeps; n<(p4est_locidx_t) nodes->indep_nodes.elem_count; ++n)
  {
#ifdef P4_TO_P8
    Point3 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes), node_z_fr_n(n, p4est, nodes));
#else
    Point2 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes));
#endif

    /* first check if the node is a voronoi point */
    for(unsigned int m=0; m<grid2voro[n].size(); ++m)
    {
      if(((pn-voro_seeds[grid2voro[n][m]]).norm_L2()< sqrt(SQR(xyz_max[0]-xyz_min[0])+SQR(xyz_max[1]-xyz_min[1])
                                                     #ifdef P4_TO_P8
                                                           +SQR(xyz_max[2]-xyz_min[2])
                                                     #endif
                                                           )*EPS) && (grid2voro[n][m] < num_local_voro))
        voro_seeds_that_are_ghost_grid_nodes[p4est->mpirank]++;
      else if(grid2voro[n][m] < num_local_voro)
        independent_voro_seeds_that_are_close_to_ghost_grid_nodes[p4est->mpirank]++;
    }
  }

  MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED, &voro_seeds_that_are_local_grid_nodes[0], 1, MPI_UNSIGNED, p4est->mpicomm);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED, &independent_voro_seeds_that_are_close_to_local_grid_nodes[0], 1, MPI_UNSIGNED, p4est->mpicomm);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED, &voro_seeds_that_are_ghost_grid_nodes[0], 1, MPI_UNSIGNED, p4est->mpicomm);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED, &independent_voro_seeds_that_are_close_to_ghost_grid_nodes[0], 1, MPI_UNSIGNED, p4est->mpicomm);

  /* write voronoi stats */
  if(p4est->mpirank==0)
  {
    FILE *f = fopen(path, "w");
    fprintf(f, "%% rank  |  total number of Voronoi seeds locally owned  |  ... that are ...  |  local grid nodes  |  close to a local grid node  |  ghost grid nodes  |  close to a ghost grid node  |  validity check  \n");
    for(int i=0; i<p4est->mpisize; ++i)
      fprintf(f, "%8d %45d                    %22u %30u %20u %30u     %s\n",
              i,
              voro_global_offset[i+1]-voro_global_offset[i],
          voro_seeds_that_are_local_grid_nodes[i],
          independent_voro_seeds_that_are_close_to_local_grid_nodes[i],
          voro_seeds_that_are_ghost_grid_nodes[i],
          independent_voro_seeds_that_are_close_to_ghost_grid_nodes[i],
          ((voro_seeds_that_are_ghost_grid_nodes[i] == (unsigned int) 0) && ((unsigned int) (voro_global_offset[i+1]-voro_global_offset[i]) == voro_seeds_that_are_local_grid_nodes[i] + independent_voro_seeds_that_are_close_to_local_grid_nodes[i] + voro_seeds_that_are_ghost_grid_nodes[i] + independent_voro_seeds_that_are_close_to_ghost_grid_nodes[i]))? "IT'S ALRIGHT" : "PROBLEM");
    fclose(f);
  }
}



void my_p4est_poisson_jump_nodes_voronoi_t::print_voronoi_VTK(const char* path) const
{
#ifdef P4_TO_P8
  std::vector<Voronoi3D> voro(num_local_voro);
#else
  std::vector<Voronoi2D> voro(num_local_voro);
#endif
  for(unsigned int n=0; n<num_local_voro; ++n)
    compute_voronoi_cell(n, voro[n]);

  char name[1000];
  sprintf(name, "%s_%d.vtk", path, p4est->mpirank);


#ifdef P4_TO_P8
  bool periodic[P4EST_DIM] = {false, false, false};
  Voronoi3D::print_VTK_format(voro, name, xyz_min, xyz_max, periodic);
#else
  Voronoi2D::print_VTK_format(voro, name);
#endif
}


void my_p4est_poisson_jump_nodes_voronoi_t::check_voronoi_partition() const
{
  PetscErrorCode ierr;
  ierr = PetscPrintf(p4est->mpicomm, "Checking partition ...\n"); CHKERRXX(ierr);
#ifdef P4_TO_P8
  std::vector<Voronoi3D> voro(num_local_voro);
  const std::vector<ngbd3Dseed> *points;
  const std::vector<ngbd3Dseed> *pts;
#else
  std::vector<Voronoi2D> voro(num_local_voro);
  const std::vector<ngbd2Dseed> *points;
  const std::vector<ngbd2Dseed> *pts;
#endif

  std::vector< int > send_to(p4est->mpisize, 0);
  std::vector< std::vector<check_comm_t> > buff_send(p4est->mpisize);

  for(unsigned int n=0; n<num_local_voro; ++n)
    compute_voronoi_cell(n, voro[n]);

  int nb_bad = 0;
  for(unsigned int n=0; n<num_local_voro; ++n)
  {
    voro[n].get_neighbor_seeds(points);

    for(unsigned int m=0; m<points->size(); ++m)
    {
      if((*points)[m].n>=0)
      {
        if((*points)[m].n<(int) num_local_voro)
        {
          voro[(*points)[m].n].get_neighbor_seeds(pts);
          bool ok = false;
          for(unsigned int k=0; k<pts->size(); ++k)
          {
            if((*pts)[k].n==(int) n)
            {
              ok = true;
              break;
            }
          }

          if(ok==false)
          {
#ifdef P4_TO_P8
            std::cout << p4est->mpirank << " found bad voronoi cell for point # " << n << " : " << (*points)[m].n << ", \t surface = " << (*points)[m].s <<  ", \t Centered on : " << voro[n].get_center_point();
#else
            std::cout << p4est->mpirank << " found bad voronoi cell for point # " << n << " : " << (*points)[m].n << ", \t Centered on : " << voro[n].get_center_point();
#endif
            nb_bad++;
          }
        }
        else
        {
          check_comm_t tmp;
          tmp.n = n;
          tmp.k = voro_ghost_local_num[(*points)[m].n-num_local_voro];
          send_to[voro_ghost_rank[(*points)[m].n-num_local_voro]] = 1;
          buff_send[voro_ghost_rank[(*points)[m].n-num_local_voro]].push_back(tmp);
        }
      }
    }
  }

  /* initiate communication */
  std::vector<MPI_Request> requests;
  for(int r=0; r<p4est->mpisize; ++r)
  {
    if(send_to[r] == 1)
    {
      MPI_Request req;
      MPI_Isend(&buff_send[r][0], buff_send[r].size()*sizeof(check_comm_t), MPI_BYTE, r, 8, p4est->mpicomm, &req);
      requests.push_back(req);
    }
  }

  /* now receive */
  // get the number of messages to receive
  int nb_recv;
  std::vector<int> nb_int_per_proc(p4est->mpisize, 1);
  int mpiret = MPI_Reduce_scatter(&send_to[0], &nb_recv, &nb_int_per_proc[0], MPI_INT, MPI_SUM, p4est->mpicomm); SC_CHECK_MPI(mpiret);
  while(nb_recv>0)
  {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 8, p4est->mpicomm, &status);
    int vec_size;
    MPI_Get_count(&status, MPI_BYTE, &vec_size);
    vec_size /= sizeof(check_comm_t);

    std::vector<check_comm_t> buff_recv(vec_size);
    MPI_Recv(&buff_recv[0], vec_size*sizeof(check_comm_t), MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, p4est->mpicomm, &status);

    for(unsigned int s=0; s<buff_recv.size(); ++s)
    {
      int local_idx = buff_recv[s].k;
      int ghost_idx = buff_recv[s].n;

      if(local_idx<0 || local_idx>=(int) num_local_voro)
        throw std::invalid_argument("my_p4est_poisson_jump_nodes_voronoi_t->check_voronoi_partition: asked to check a non local point or a wall.");

      voro[local_idx].get_neighbor_seeds(pts);
      bool ok = false;
      for(unsigned int k=0; k<pts->size(); ++k)
      {
        if((*pts)[k].n>=(int) num_local_voro && voro_ghost_local_num[(*pts)[k].n-num_local_voro]==ghost_idx)
        {
          ok = true;
          continue;
        }
      }

      if(ok==false)
      {
        std::cout << p4est->mpirank << " found bad ghost voronoi cell for point # " << local_idx << " : " << ghost_idx << ", \t Centered on : " << voro[local_idx].get_center_point();
        nb_bad++;
      }
    }

    nb_recv--;
  }

  MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);

  MPI_Allreduce(MPI_IN_PLACE, (void*) &nb_bad, 1, MPI_INT, MPI_SUM, p4est->mpicomm);

  if(nb_bad==0) { ierr = PetscPrintf(p4est->mpicomm, "Partition is good.\n"); CHKERRXX(ierr); }
  else          { ierr = PetscPrintf(p4est->mpicomm, "Partition is NOT good, %d problem found.\n", nb_bad); CHKERRXX(ierr); }
}
