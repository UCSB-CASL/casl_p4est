#include <src/my_p4est_poisson_jump_voronoi_block.h>
#include <src/my_p4est_refine_coarsen.h>

#include <algorithm>
#include <cassert>

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

my_p4est_poisson_jump_voronoi_block_t::my_p4est_poisson_jump_voronoi_block_t(
    int block_size,
    const my_p4est_node_neighbors_t *node_neighbors,
    const my_p4est_cell_neighbors_t *cell_neighbors)
  : block_size(block_size), ngbd_n(node_neighbors),  ngbd_c(cell_neighbors), myb(node_neighbors->myb),
    p4est(node_neighbors->p4est), ghost(node_neighbors->ghost), nodes(node_neighbors->nodes),
    phi(NULL), rhs(NULL), sol_voro(NULL),
    voro_global_offset(p4est->mpisize),
    interp_phi(node_neighbors),
    rhs_m(block_size),
    rhs_p(block_size),
    A(PETSC_NULL), A_null_space(PETSC_NULL), ksp(PETSC_NULL),
    is_voronoi_partition_constructed(false), is_matrix_computed(false), matrix_has_nullspace(false)
{
  // set up the KSP solver
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);

  // compute grid parameters
  // NOTE: Assuming all trees are of the same size [0, 1]^d
  p4est_xyz_min(p4est, xyz_min);
  p4est_xyz_max(p4est, xyz_max);

  xmin = xyz_min[0], ymin = xyz_min[1];
  xmax = xyz_max[0], ymax = xyz_max[1];
#ifdef P4_TO_P8
  zmin = xyz_min[2];
  zmax = xyz_max[2];
#endif

  p4est_dxyz_min(p4est, dxyz_min_);

  dx_min = dxyz_min_[0], dy_min = dxyz_min_[1];

#ifdef P4_TO_P8
  dz_min = dxyz_min_[2];
  d_min = MIN(dx_min, dy_min, dz_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min + dz_min*dz_min);
#else
  d_min = MIN(dx_min, dy_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min);
#endif
}


my_p4est_poisson_jump_voronoi_block_t::~my_p4est_poisson_jump_voronoi_block_t()
{
  if(A            != PETSC_NULL) { ierr = MatDestroy(A);                     CHKERRXX(ierr); }
  if(A_null_space != PETSC_NULL) { ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr); }
  if(ksp          != PETSC_NULL) { ierr = KSPDestroy(ksp);                   CHKERRXX(ierr); }
  if(rhs          != PETSC_NULL) { ierr = VecDestroy(rhs);                   CHKERRXX(ierr); }
  for (int i=0; i<block_size; i++) {
    delete rhs_m[i];
    delete rhs_p[i];
  }
}


PetscErrorCode my_p4est_poisson_jump_voronoi_block_t::VecCreateGhostVoronoiRhs()
{
  PetscErrorCode ierr = 0;
  PetscInt num_local = num_local_voro;
  PetscInt num_global = voro_global_offset[p4est->mpisize];

  std::vector<PetscInt> ghost_voro(voro_points.size() - num_local, 0);

  for (size_t i = 0; i<ghost_voro.size(); ++i)
  {
    ghost_voro[i] = voro_ghost_local_num[i] + voro_global_offset[voro_ghost_rank[i]];
  }

  if(rhs!=PETSC_NULL) VecDestroy(rhs);

  ierr = VecCreateGhostBlock(p4est->mpicomm,
                             block_size, block_size*num_local_voro, block_size*num_global,
                             ghost_voro.size(), (const PetscInt*)&ghost_voro[0], &rhs); CHKERRQ(ierr);
  ierr = VecSetFromOptions(rhs); CHKERRQ(ierr);

  return ierr;
}

void my_p4est_poisson_jump_voronoi_block_t::inverse(double* mue, double* mue_inv)
{
  if (block_size == 1) {
    mue_inv[0] = 1.0 / mue[0];
    return;

  } else if (block_size == 2) {
    double det = mue[0]*mue[3] - mue[1]*mue[2];
    mue_inv[0] =  mue[3]/det;
    mue_inv[1] = -mue[1]/det;
    mue_inv[2] = -mue[2]/det;
    mue_inv[3] =  mue[0]/det;

    return;
  } else {
    throw std::invalid_argument("The block solver currently only works with 2x2 system\n");
  }
}

void my_p4est_poisson_jump_voronoi_block_t::matmult(double* mue_1, double* mue_2, double* mue)
{
  for (int i=0; i<block_size; i++) {
    for (int j=0; j<block_size; j++) {
      mue[i*block_size+j] = 0;
      for (int k=0; k<block_size; k++) {
        mue[i*block_size + j] += mue_1[i*block_size+k]*mue_2[k*block_size+j];
      }
    }
  }
}

void my_p4est_poisson_jump_voronoi_block_t::set_phi(Vec phi)
{
  this->phi = phi;
  interp_phi.set_input(phi, linear);
}

void my_p4est_poisson_jump_voronoi_block_t::set_rhs(Vec rhs_m[], Vec rhs_p[])
{
  for (int i = 0; i<block_size; i++) {
    this->rhs_m[i] = new my_p4est_interpolation_nodes_t(ngbd_n);
    this->rhs_m[i]->set_input(rhs_m[i], linear);

    this->rhs_p[i] = new my_p4est_interpolation_nodes_t(ngbd_n);
    this->rhs_p[i]->set_input(rhs_p[i], linear);
  }
}

void my_p4est_poisson_jump_voronoi_block_t::set_diagonal(vector<vector<cf_t*> >& add)
{
  this->add = add;
}

void my_p4est_poisson_jump_voronoi_block_t::set_bc(vector<bc_t> &bc)
{
  this->bc = bc;
  is_matrix_computed = false;
}

void my_p4est_poisson_jump_voronoi_block_t::set_mu(vector<vector<cf_t*> >& mu_m, vector<vector<cf_t*> > &mu_p)
{
  this->mu_m = mu_m;
  this->mu_p = mu_p;
}

void my_p4est_poisson_jump_voronoi_block_t::set_u_jump(vector<cf_t*> &u_jump)
{
  this->u_jump = u_jump;
}

void my_p4est_poisson_jump_voronoi_block_t::set_mu_grad_u_jump(vector<cf_t*> &mu_grad_u_jump)
{
  this->mu_grad_u_jump = mu_grad_u_jump;
}

void my_p4est_poisson_jump_voronoi_block_t::solve(Vec solution[], bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_solve, A, rhs, ksp, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  for (int i=0; i<block_size; i++) {
    PetscInt sol_size;
    ierr = VecGetLocalSize(solution[i], &sol_size); CHKERRXX(ierr);
    if (sol_size != nodes->num_owned_indeps){
      std::ostringstream oss;
      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as num_owned_indeps"
          << "solution.local_size = " << sol_size << " nodes->num_owned_indeps = " << nodes->num_owned_indeps << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
#endif

  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);  
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  /* first compute the voronoi partition */
  if(!is_voronoi_partition_constructed)
  {
    is_voronoi_partition_constructed = true;
//    ierr = PetscPrintf(p4est->mpicomm, "Computing voronoi points ...\n"); CHKERRXX(ierr);
    compute_voronoi_points();
//    ierr = PetscPrintf(p4est->mpicomm, "Done computing voronoi points.\n"); CHKERRXX(ierr);
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
    // setup_negative_laplace_rhsvec();
    setup_linear_system();
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

  double res;
  ierr = KSPGetResidualNorm(ksp, &res); CHKERRXX(ierr);


   ierr = PetscPrintf(p4est->mpicomm, "Solving linear system. Residual: %f \n", res); CHKERRXX(ierr);
  //ierr = PetscPrintf(p4est->mpicomm, "Solving linear system ...\n"); CHKERRXX(ierr);
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_KSPSolve, ksp, rhs, sol_voro, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, sol_voro); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_PoissonSolverNodeBasedJump_KSPSolve, ksp, rhs, sol_voro, 0); CHKERRXX(ierr);
  //ierr = PetscPrintf(p4est->mpicomm, "Done solving linear system.\n"); CHKERRXX(ierr);
  ierr = KSPGetResidualNorm(ksp, &res); CHKERRXX(ierr);
   ierr = PetscPrintf(p4est->mpicomm, "Done solving linear system. Residual: %f \n", res); CHKERRXX(ierr);

  /* update ghosts */
  ierr = VecGhostUpdateBegin(sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* interpolate the solution back onto the original mesh */
  interpolate_solution_from_voronoi_to_tree(solution);

  ierr = VecDestroy(sol_voro); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
}


void my_p4est_poisson_jump_voronoi_block_t::compute_voronoi_points()
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_compute_voronoi_points, 0, 0, 0, 0); CHKERRXX(ierr);

  if(grid2voro.size()!=0)
  {
    for(unsigned int n=0; n<grid2voro.size(); ++n)
      grid2voro[n].clear();
  }
  grid2voro.resize(nodes->indep_nodes.elem_count);

  voro_points.clear();

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  std::vector<p4est_locidx_t> marked_nodes;

  /* find the projected points associated to shared nodes
   * if a projected point is shared, all larger rank are informed.
   * The goal here is to avoid building two close projected points at a processor boundary
   * and to have a consistent partition across processes
   */
  std::vector< std::vector<added_point_t> > buff_shared_added_points_send(p4est->mpisize);
  std::vector< std::vector<added_point_t> > buff_shared_added_points_recv(p4est->mpisize);
  std::vector<bool> send_shared_to(p4est->mpisize, false);

  for(size_t l=0; l<ngbd_n->get_layer_size(); ++l)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(l);
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
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
      send_shared_to[sharers[s]] = true;

#ifdef P4_TO_P8
    double p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p;
    (*ngbd_n).get_neighbors(n).ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p);
    if(p_000*p_m00<=0 || p_000*p_p00<=0 || p_000*p_0m0<=0 || p_000*p_0p0<=0 || p_000*p_00m<=0 || p_000*p_00p<=0)
#else
    double p_00, p_m0, p_p0, p_0m, p_0p;
    (*ngbd_n).get_neighbors(n).ngbd_with_quadratic_interpolation(phi_p, p_00, p_m0, p_p0, p_0m, p_0p);
    if(p_00*p_m0<=0 || p_00*p_p0<=0 || p_00*p_0m<=0 || p_00*p_0p<=0)
#endif
    {
      double d = phi_p[n];
#ifdef P4_TO_P8
      Point3 dp((*ngbd_n).get_neighbors(n).dx_central(phi_p), (*ngbd_n).get_neighbors(n).dy_central(phi_p), (*ngbd_n).get_neighbors(n).dz_central(phi_p));
#else
      // FIXME: These kind of operations would be expensive if neighbors is not initialized!
      Point2 dp((*ngbd_n).get_neighbors(n).dx_central(phi_p), (*ngbd_n).get_neighbors(n).dy_central(phi_p));
#endif
      dp /= dp.norm_L2();
      double xn = node_x_fr_n(n, p4est, nodes);
      double yn = node_y_fr_n(n, p4est, nodes);
      added_point_t added_point_n;
      added_point_n.x = xn-d*dp.x;
      added_point_n.y = yn-d*dp.y;
      added_point_n.dx = dp.x;
      added_point_n.dy = dp.y;
#ifdef P4_TO_P8
      double zn = node_z_fr_n(n, p4est, nodes);
      added_point_n.z = zn-d*dp.z;
      added_point_n.dz = dp.z;
#endif

      for(size_t s=0; s<num_sharers; ++s)
      {
        buff_shared_added_points_send[sharers[s]].push_back(added_point_n);
      }
      buff_shared_added_points_recv[p4est->mpirank].push_back(added_point_n);
    }
    else
      marked_nodes.push_back(n);
  }

  /* send the shared points to the corresponding neighbors ranks
   * note that some messages have a size 0 since the processes can't know who is going to send them data
   * in order to find that out, one needs to call ngbd_with_quadratic_interpolation on ghost nodes...
   */
  std::vector<MPI_Request> req_shared_added_points;
  for(int r=0; r<p4est->mpisize; ++r)
  {
    if(send_shared_to[r]==true)
    {
      MPI_Request req;
      MPI_Isend(&buff_shared_added_points_send[r][0], buff_shared_added_points_send[r].size()*sizeof(added_point_t), MPI_BYTE, r, 4, p4est->mpicomm, &req);
      req_shared_added_points.push_back(req);
    }
  }

  /* add the nodes that are actual voronoi points (not close to interface)
   * to the list of voronoi points
   */
  /* layer nodes first */
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
    grid2voro[n].push_back(voro_points.size());
    voro_points.push_back(p);
  }

  /* now local nodes */
  marked_nodes.clear();
  for(size_t l=0; l<ngbd_n->get_local_size(); ++l)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(l);
#ifdef P4_TO_P8
    double p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p;
    (*ngbd_n).get_neighbors(n).ngbd_with_quadratic_interpolation(phi_p, p_000, p_m00, p_p00, p_0m0, p_0p0, p_00m, p_00p);
    if(!(p_000*p_m00<=0 || p_000*p_p00<=0 || p_000*p_0m0<=0 || p_000*p_0p0<=0 || p_000*p_00m<=0 || p_000*p_00p<=0))
#else
    double p_00, p_m0, p_p0, p_0m, p_0p;
    (*ngbd_n).get_neighbors(n).ngbd_with_quadratic_interpolation(phi_p, p_00, p_m0, p_p0, p_0m, p_0p);
    if(!(p_00*p_m0<=0 || p_00*p_p0<=0 || p_00*p_0m<=0 || p_00*p_0p<=0))
#endif
    {
      double xn = node_x_fr_n(n, p4est, nodes);
      double yn = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double zn = node_z_fr_n(n, p4est, nodes);
      Point3 p(xn, yn, zn);
#else
      Point2 p(xn, yn);
#endif
      grid2voro[n].push_back(voro_points.size());
      voro_points.push_back(p);
    }
    else
      marked_nodes.push_back(n);
  }

  /* compute how many messages we are expecting to receive */
  std::vector<bool> recv_shared_fr(p4est->mpisize, false);
  for(size_t n=nodes->num_owned_indeps; n<nodes->indep_nodes.elem_count; ++n)
    recv_shared_fr[nodes->nonlocal_ranks[n-nodes->num_owned_indeps]] = true;

  int nb_rcv = 0;
  for(int r=0; r<p4est->mpisize; ++r)
    if(recv_shared_fr[r]==true) nb_rcv++;

  /* now receive the points */
  while(nb_rcv>0)
  {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 4, p4est->mpicomm, &status);
    int vec_size;
    MPI_Get_count(&status, MPI_BYTE, &vec_size);
    vec_size /= sizeof(added_point_t);

    buff_shared_added_points_recv[status.MPI_SOURCE].resize(vec_size);
    MPI_Recv(&buff_shared_added_points_recv[status.MPI_SOURCE][0], vec_size*sizeof(added_point_t), MPI_BYTE, status.MPI_SOURCE, 4, p4est->mpicomm, &status);

    nb_rcv--;
  }

  /* now add the points to the list of projected points */
#ifdef P4_TO_P8
  std::vector<Point3> added_points;
  std::vector<Point3> added_points_grad;
#else
  std::vector<Point2> added_points;
  std::vector<Point2> added_points_grad;
#endif
  for(int r=0; r<p4est->mpisize; ++r)
  {
    for(unsigned int m=0; m<buff_shared_added_points_recv[r].size(); ++m)
    {
#ifdef P4_TO_P8
      Point3 p(buff_shared_added_points_recv[r][m].x, buff_shared_added_points_recv[r][m].y, buff_shared_added_points_recv[r][m].z);
#else
      Point2 p(buff_shared_added_points_recv[r][m].x, buff_shared_added_points_recv[r][m].y);
#endif

      bool already_added = false;
      for(unsigned int k=0; k<added_points.size(); ++k)
      {
        if((p-added_points[k]).norm_L2() < diag_min/10)
        {
          already_added = true;
          break;
        }
      }

      if(!already_added)
      {
        added_points.push_back(p);
#ifdef P4_TO_P8
        Point3 dp(buff_shared_added_points_recv[r][m].dx, buff_shared_added_points_recv[r][m].dy, buff_shared_added_points_recv[r][m].dz);
#else
        Point2 dp(buff_shared_added_points_recv[r][m].dx, buff_shared_added_points_recv[r][m].dy);
#endif
        added_points_grad.push_back(dp);
      }
    }

    //    buff_shared_added_points_send[r].clear();
    buff_shared_added_points_recv[r].clear();
  }

  //  buff_shared_added_points_send.clear();
  buff_shared_added_points_recv.clear();

  /* add the local points to the list of projected points */
  for(size_t i=0; i<marked_nodes.size(); ++i)
  {
    p4est_locidx_t n = marked_nodes[i];

    double xn = node_x_fr_n(n, p4est, nodes);
    double yn = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double zn = node_z_fr_n(n, p4est, nodes);
    Point3 dp((*ngbd_n).get_neighbors(n).dx_central(phi_p), (*ngbd_n).get_neighbors(n).dy_central(phi_p), (*ngbd_n).get_neighbors(n).dz_central(phi_p));
#else
    Point2 dp((*ngbd_n).get_neighbors(n).dx_central(phi_p), (*ngbd_n).get_neighbors(n).dy_central(phi_p));
#endif

    double d = phi_p[n];
    dp /= dp.norm_L2();

#ifdef P4_TO_P8
    Point3 p_proj(xn-d*dp.x, yn-d*dp.y, zn-d*dp.z);
#else
    Point2 p_proj(xn-d*dp.x, yn-d*dp.y);
#endif

    bool already_added = false;
    for(unsigned int m=0; m<added_points.size(); ++m)
    {
      if((p_proj-added_points[m]).norm_L2() < diag_min/10)
      {
        already_added = true;
        break;
      }
    }

    if(!already_added)
    {
      added_points.push_back(p_proj);
      added_points_grad.push_back(dp);
    }
  }

  /* finally build the voronoi points from the list of projected points */
  double band = diag_min/10;
  for(unsigned int n=0; n<added_points.size(); ++n)
  {
#ifdef P4_TO_P8
    Point3 p_proj = added_points[n];
    Point3 dp = added_points_grad[n];
#else
    Point2 p_proj = added_points[n];
    Point2 dp = added_points_grad[n];
#endif

    /* add first point */
    double xyz1 [] =
    {
      std::min(xyz_max[0], std::max(xyz_min[0], p_proj.x + band*dp.x)),
      std::min(xyz_max[1], std::max(xyz_min[1], p_proj.y + band*dp.y))
  #ifdef P4_TO_P8
      , std::min(xyz_max[2], std::max(xyz_min[2], p_proj.z + band*dp.z))
  #endif
    };

    p4est_quadrant_t quad;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz1, quad, remote_matches);

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

      p4est_locidx_t node = -1;
#ifdef P4_TO_P8
      if     (xyz1[0]<=qx && xyz1[1]<=qy && xyz1[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
      else if(xyz1[0]<=qx && xyz1[1]<=qy && xyz1[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmp];
      else if(xyz1[0]<=qx && xyz1[1]> qy && xyz1[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
      else if(xyz1[0]<=qx && xyz1[1]> qy && xyz1[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpp];
      else if(xyz1[0]> qx && xyz1[1]<=qy && xyz1[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
      else if(xyz1[0]> qx && xyz1[1]<=qy && xyz1[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmp];
      else if(xyz1[0]> qx && xyz1[1]> qy && xyz1[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];
      else if(xyz1[0]> qx && xyz1[1]> qy && xyz1[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppp];
#else
      if     (xyz1[0]<=qx && xyz1[1]<=qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
      else if(xyz1[0]<=qx && xyz1[1]> qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
      else if(xyz1[0]> qx && xyz1[1]<=qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
      else if(xyz1[0]> qx && xyz1[1]> qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];
#endif

      grid2voro[node].push_back(voro_points.size());
#ifdef P4_TO_P8
      Point3 p(xyz1[0], xyz1[1], xyz1[2]);
#else
      Point2 p(xyz1[0], xyz1[1]);
#endif
      voro_points.push_back(p);
    }

    /* add second point */
    double xyz2 [] =
    {
      std::min(xyz_max[0], std::max(xyz_min[0], p_proj.x - band*dp.x)),
      std::min(xyz_max[1], std::max(xyz_min[1], p_proj.y - band*dp.y))
  #ifdef P4_TO_P8
      , std::min(xyz_max[2], std::max(xyz_min[2], p_proj.z - band*dp.z))
  #endif
    };

    remote_matches.clear();
    rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz2, quad, remote_matches);

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

      p4est_locidx_t node = -1;
#ifdef P4_TO_P8
      if     (xyz2[0]<=qx && xyz2[1]<=qy && xyz2[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
      else if(xyz2[0]<=qx && xyz2[1]<=qy && xyz2[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmp];
      else if(xyz2[0]<=qx && xyz2[1]> qy && xyz2[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
      else if(xyz2[0]<=qx && xyz2[1]> qy && xyz2[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpp];
      else if(xyz2[0]> qx && xyz2[1]<=qy && xyz2[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
      else if(xyz2[0]> qx && xyz2[1]<=qy && xyz2[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmp];
      else if(xyz2[0]> qx && xyz2[1]> qy && xyz2[2]<=qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];
      else if(xyz2[0]> qx && xyz2[1]> qy && xyz2[2]> qz) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppp];
#else
      if     (xyz2[0]<=qx && xyz2[1]<=qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
      else if(xyz2[0]<=qx && xyz2[1]> qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
      else if(xyz2[0]> qx && xyz2[1]<=qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
      else if(xyz2[0]> qx && xyz2[1]> qy) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];
#endif

      grid2voro[node].push_back(voro_points.size());
#ifdef P4_TO_P8
      Point3 p(xyz2[0], xyz2[1], xyz2[2]);
#else
      Point2 p(xyz2[0], xyz2[1]);
#endif
      voro_points.push_back(p);
    }
  }

  added_points.clear();
  added_points_grad.clear();

  /* prepare the buffer to send shared local voro points */
  std::vector< std::vector<voro_comm_t> > buff_send_points(p4est->mpisize);
  std::vector<bool> send_to(p4est->mpisize, false);

  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
    /* if the node is shared, add to corresponding buffer
     * note that we are sending empty messages to some processes, this is
     * because checking who is going to send a message requires a communication ...
     * so we just send a message to all possible processes, even if the message is empty
     */
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
        send_to[sharers[s]] = true;

        for(unsigned int m=0; m<grid2voro[n].size(); ++m)
        {
          voro_comm_t v;
          v.local_num = grid2voro[n][m];
          v.x = voro_points[grid2voro[n][m]].x;
          v.y = voro_points[grid2voro[n][m]].y;
#ifdef P4_TO_P8
          v.z = voro_points[grid2voro[n][m]].z;
#endif
          buff_send_points[sharers[s]].push_back(v);
        }
      }
    }
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  /* send the data to remote processes */
  std::vector<MPI_Request> req_send_points;
  for(int r=0; r<p4est->mpisize; ++r)
  {
    if(send_to[r]==true)
    {
      MPI_Request req;
      MPI_Isend((void*)&buff_send_points[r][0], buff_send_points[r].size()*sizeof(voro_comm_t), MPI_BYTE, r, 2, p4est->mpicomm, &req);
      req_send_points.push_back(req);
    }
  }

  /* get local number of voronoi points for every processor */
  num_local_voro = voro_points.size();
  voro_global_offset[p4est->mpirank] = num_local_voro;
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &voro_global_offset[0], 1, MPI_INT, p4est->mpicomm);
  for(int r=1; r<p4est->mpisize; ++r)
  {
    voro_global_offset[r] += voro_global_offset[r-1];
  }

  voro_global_offset.insert(voro_global_offset.begin(), 0);
//  ierr = PetscPrintf(p4est->mpicomm, "Number of voronoi points : %d\n", voro_global_offset[p4est->mpisize]);

  /* initialize the buffer to receive remote points */
  std::vector<bool> recv_fr(p4est->mpisize);
  for(size_t n=0; n<nodes->indep_nodes.elem_count; ++n)
  {
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
        recv_fr[sharers[s]] = true;
      }
    }
  }

  nb_rcv = 0;
  for(int r=0; r<p4est->mpisize; ++r)
    if(recv_fr[r]) nb_rcv++;

  /* now receive the data */
  while(nb_rcv>0)
  {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, 2, p4est->mpicomm, &status);

    int nb_points;
    MPI_Get_count(&status, MPI_BYTE, &nb_points);
    nb_points /= sizeof(voro_comm_t);

    std::vector<voro_comm_t> buff_recv_points(nb_points);

    int sender_rank = status.MPI_SOURCE;
    MPI_Recv(&buff_recv_points[0], nb_points*sizeof(voro_comm_t), MPI_BYTE, sender_rank, status.MPI_TAG, p4est->mpicomm, &status);

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
        if(rank_found==p4est->mpirank) quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;
        else                           quad_idx = quad.p.piggy3.local_num + p4est->local_num_quadrants;

        double qx = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
        double qy = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
        double qz = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
#endif

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

        grid2voro[node].push_back(voro_points.size());
#ifdef P4_TO_P8
        Point3 p(xyz[0], xyz[1], xyz[2]);
#else
        Point2 p(xyz[0], xyz[1]);
#endif
        voro_points.push_back(p);

        voro_ghost_local_num.push_back(buff_recv_points[n].local_num);
        voro_ghost_rank.push_back(sender_rank);
      }
    }

    nb_rcv--;
  }

  MPI_Waitall(req_shared_added_points.size(), &req_shared_added_points[0], MPI_STATUSES_IGNORE);
  MPI_Waitall(req_send_points.size(), &req_send_points[0], MPI_STATUSES_IGNORE);

  /* clear buffers */
  //  for(int r=0; r<p4est->mpisize; ++r)
  //    buff_send_points[r].clear();
  //  buff_send_points.clear();
  send_to.clear();
  recv_fr.clear();

  ierr = VecCreateGhostVoronoiRhs(); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_compute_voronoi_points, 0, 0, 0, 0); CHKERRXX(ierr);
}




#ifdef P4_TO_P8
void my_p4est_poisson_jump_voronoi_block_t::compute_voronoi_cell(unsigned int n, Voronoi3D &voro) const
#else
void my_p4est_poisson_jump_voronoi_block_t::compute_voronoi_cell(unsigned int n, Voronoi2D &voro) const
#endif
{
  PetscErrorCode ierr;
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);

  /* find the cell to which this point belongs */
#ifdef P4_TO_P8
  Point3 pc;
#else
  Point2 pc;
#endif
  pc = voro_points[n];

  double xyz [] =
  {
    pc.x,
    pc.y
  #ifdef P4_TO_P8
    , pc.z
  #endif
  };
  p4est_quadrant_t quad;
  std::vector<p4est_quadrant_t> remote_matches;
  int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);

  /* check if the point is exactly a node */
  p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
  p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);

  // FIXME: This does not work if domain size is not the same in xyz directions
  double qhx = quad_dx(p4est, &quad);
  double qx  = quad_x(p4est,  &quad);
  double qhy = quad_dy(p4est, &quad);
  double qy  = quad_y(p4est,  &quad);
#ifdef P4_TO_P8
  double qz  = quad_z(p4est,  &quad);
  double qhz = quad_dz(p4est, &quad);
#endif

#ifdef P4_TO_P8
  // FIXME: Is this correct? comprare to old voronoi solver
  voro.set_Center_Point(n,pc);
#else
  voro.set_Center_Point(pc);
#endif

  p4est_locidx_t quad_idx;
#ifdef CASL_THROWS
  if(rank_found==-1)
    throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_jump_nodes_voronoi_t->compute_voronoi_mesh: found remote quadrant.");
#endif
  if(rank_found==p4est->mpirank) quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;
  else                           quad_idx = quad.p.piggy3.local_num + p4est->local_num_quadrants;

  std::vector<p4est_locidx_t> ngbd_quads;

  /* if exactly on a grid node */
  if( (fabs(xyz[0]-(qx-0.5*qhx))<EPS || fabs(xyz[0]-(qx+0.5*qhx))<EPS) &&
      (fabs(xyz[1]-(qy-0.5*qhy))<EPS || fabs(xyz[1]-(qy+0.5*qhy))<EPS)
    #ifdef P4_TO_P8
      && (fabs(xyz[2]-(qz-0.5*qhz))<EPS || fabs(xyz[2]-(qz+0.5*qhz))<EPS)
    #endif
      )
  {
#ifdef P4_TO_P8
    int dir = (fabs(xyz[0]-(qx-0.5*qhx))<EPS ?
          (fabs(xyz[1]-(qy-0.5*qhy))<EPS ?
            (fabs(xyz[2]-(qz-0.5*qhz))<EPS ? dir::v_mmm : dir::v_mmp)
          : (fabs(xyz[2]-(qz-0.5*qhz))<EPS ? dir::v_mpm : dir::v_mpp) )
        : (fabs(xyz[1]-(qy-0.5*qhy))<EPS ?
            (fabs(xyz[2]-(qz-0.5*qhz))<EPS ? dir::v_pmm : dir::v_pmp)
          : (fabs(xyz[2]-(qz-0.5*qhz))<EPS ? dir::v_ppm : dir::v_ppp) ) );
#else
    int dir = (fabs(xyz[0]-(qx-0.5*qhx))<EPS ?
          (fabs(xyz[1]-(qy-0.5*qhy))<EPS ? dir::v_mmm : dir::v_mpm)
        : (fabs(xyz[1]-(qy-0.5*qhy))<EPS ? dir::v_pmm : dir::v_ppm) );
#endif
    p4est_locidx_t node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir];

    p4est_locidx_t quad_idx;

    std::vector<p4est_quadrant_t> tmp;
#ifdef P4_TO_P8
    for(char i=-1; i<=1; i+=2)
    {
      for(char j=-1; j<=1; j+=2)
      {
        for(char k=-1; k<=1; k+=2)
        {
          ngbd_n->find_neighbor_cell_of_node(node,  i,  j, k, quad_idx, tree_idx);
          if(quad_idx>=0)
          {
            ngbd_quads.push_back(quad_idx);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, 0, 0);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, 0, j, 0);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, 0, 0, k);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, j, 0);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, 0, k);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, 0, j, k);
            for(unsigned int m=0; m<tmp.size(); ++m)
              ngbd_quads.push_back(tmp[m].p.piggy3.local_num);
            tmp.clear();
          }
        }
      }
    }
#else
    for(char i=-1; i<=1; i+=2)
    {
      for(char j=-1; j<=1; j+=2)
      {
        ngbd_n->find_neighbor_cell_of_node(node,  i,  j, quad_idx, tree_idx);
        if(quad_idx>=0)
        {
          ngbd_quads.push_back(quad_idx);
          ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, 0);
          ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, 0, j);
          for(unsigned int m=0; m<tmp.size(); ++m)
            ngbd_quads.push_back(tmp[m].p.piggy3.local_num);
          tmp.clear();
        }
      }
    }
#endif
  }
  /* the voronoi point is not a grid node */
  else
  {
    ngbd_quads.push_back(quad_idx);

    std::vector<p4est_quadrant_t> tmp;
#ifdef P4_TO_P8
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, -1,  0,  0);
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx,  1,  0,  0);
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx,  0,  1,  0);
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx,  0, -1,  0);
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx,  0,  0, -1);
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx,  0,  0,  1);
#else
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, -1,  0);
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx,  1,  0);
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx,  0,  1);
    ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx,  0, -1);
#endif

    for(unsigned int m=0; m<tmp.size(); ++m)
      ngbd_quads.push_back(tmp[m].p.piggy3.local_num);

    p4est_locidx_t n_idx;
    p4est_locidx_t q_idx;
    p4est_topidx_t t_idx;
#ifdef P4_TO_P8
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mmm];
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1, -1, -1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mpm];
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1,  1, -1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_pmm];
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1, -1, -1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_ppm];
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1,  1, -1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);

    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mmp];
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1, -1,  1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mpp];
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1,  1,  1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_pmp];
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1, -1,  1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_ppp];
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1,  1,  1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
#else
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mmm];
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1, -1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mpm];
    ngbd_n->find_neighbor_cell_of_node(n_idx, -1,  1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_pmm];
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1, -1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
    n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_ppm];
    ngbd_n->find_neighbor_cell_of_node(n_idx,  1,  1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
#endif

    std::vector<p4est_quadrant_t> tmp2;
    for(unsigned int k=0; k<tmp.size(); ++k)
    {
#ifdef P4_TO_P8
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree, -1,  0,  0);
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree,  1,  0,  0);
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree,  0, -1,  0);
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree,  0,  1,  0);
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree,  0,  0, -1);
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree,  0,  0,  1);
#else
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree, -1,  0);
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree,  1,  0);
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree,  0, -1);
      ngbd_c->find_neighbor_cells_of_cell(tmp2, tmp[k].p.piggy3.local_num, tmp[k].p.piggy3.which_tree,  0,  1);
#endif
      for(unsigned int l=0; l<tmp2.size(); ++l)
        ngbd_quads.push_back(tmp2[l].p.piggy3.local_num);
      tmp2.clear();
    }
  }

  /* now create the list of nodes */
  for(unsigned int k=0; k<ngbd_quads.size(); ++k)
  {
    for(int dir=0; dir<P4EST_CHILDREN; ++dir)
    {
      p4est_locidx_t n_idx = nodes->local_nodes[P4EST_CHILDREN*ngbd_quads[k] + dir];
      for(unsigned int m=0; m<grid2voro[n_idx].size(); ++m)
      {
        if(grid2voro[n_idx][m] != n)
        {
#ifdef P4_TO_P8
          Point3 pm = voro_points[grid2voro[n_idx][m]];
          voro.push(grid2voro[n_idx][m], pm.x, pm.y, pm.z);
#else
          Point2 pm = voro_points[grid2voro[n_idx][m]];
          voro.push(grid2voro[n_idx][m], pm.x, pm.y);
#endif
        }
      }
    }
  }

  /* add the walls */
#ifdef P4_TO_P8
  if(is_quad_xmWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_m00, pc.x-MAX(EPS, 2*(pc.x-xmin)), pc.y, pc.z);
  if(is_quad_xpWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_p00, pc.x+MAX(EPS, 2*(xmax-pc.x)), pc.y, pc.z);
  if(is_quad_ymWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_0m0, pc.x, pc.y-MAX(EPS, 2*(pc.y-ymin)), pc.z);
  if(is_quad_ypWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_0p0, pc.x, pc.y+MAX(EPS, 2*(ymax-pc.y)), pc.z);
  if(is_quad_zmWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_00m, pc.x, pc.y, pc.z-MAX(EPS, 2*(pc.z-zmin)));
  if(is_quad_zpWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_00p, pc.x, pc.y, pc.z+MAX(EPS, 2*(zmax-pc.z)));
#else
  if(is_quad_xmWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_m00, pc.x-MAX(EPS, 2*(pc.x-xmin)), pc.y );
  if(is_quad_xpWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_p00, pc.x+MAX(EPS, 2*(xmax-pc.x)), pc.y );
  if(is_quad_ymWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_0m0, pc.x, pc.y-MAX(EPS, 2*(pc.y-ymin)));
  if(is_quad_ypWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_0p0, pc.x, pc.y+MAX(EPS, 2*(ymax-pc.y)));
#endif

  /* finally, construct the partition */
#ifdef P4_TO_P8
  double xyz_min [] = {xmin, ymin, zmin};
  double xyz_max [] = {xmax, ymax, zmax};
  bool periodic []  = {false, false, false};
  voro.construct_Partition(xyz_min, xyz_max, periodic);
#else
  voro.construct_Partition();
#endif

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);
}



void my_p4est_poisson_jump_voronoi_block_t::setup_linear_system()
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  typedef struct entry
  {
    double val;
    PetscInt n;
  } entry_t;

  std::vector< std::vector<entry_t> > matrix_entries(block_size*num_local_voro);
  std::vector<PetscInt> d_nnz(block_size*num_local_voro, block_size), o_nnz(block_size*num_local_voro, 0);

  for(unsigned int n=0; n<num_local_voro; ++n) {
    PetscInt global_n_idx = n+voro_global_offset[p4est->mpirank];

#ifdef P4_TO_P8
    Point3 pc = voro_points[n];
#else
    Point2 pc = voro_points[n];
#endif

#ifdef P4_TO_P8
    Voronoi3D voro;
#else
    Voronoi2D voro;
#endif
    compute_voronoi_cell(n, voro);

#ifdef P4_TO_P8
    const std::vector<Voronoi3DPoint> *points;
#else
    const std::vector<Point2> *partition;
    const std::vector<Voronoi2DPoint> *points;
    voro.get_Partition(partition);
#endif
    voro.get_Points(points);
#ifndef P4_TO_P8
    voro.compute_volume();
#endif
    double volume = voro.get_volume();

#ifdef P4_TO_P8
    double phi_n = interp_phi(pc.x, pc.y, pc.z);
#else
    double phi_n = interp_phi(pc.x, pc.y);
#endif

    double mue_n[block_size][block_size],
        mue_l[block_size][block_size],
        mue_h[block_size][block_size],
        mue_tmp[block_size][block_size],
        mue_inv[block_size][block_size];

    // compute mue
    for (int bi=0; bi<block_size; bi++) {
      for (int bj=0; bj<block_size; bj++) {
        if (phi_n < 0) {
  #ifdef P4_TO_P8
          mue_n[bi][bj] = (*mu_m[bi][bj])(pc.x, pc.y, pc.z);
  #else
          mue_n[bi][bj] = (*mu_m[bi][bj])(pc.x, pc.y);
  #endif
        } else {
  #ifdef P4_TO_P8
          mue_n[bi][bj] = (*mu_p[bi][bj])(pc.x, pc.y, pc.z);
  #else
          mue_n[bi][bj] = (*mu_p[bi][bj])(pc.x, pc.y);
  #endif
        }
      }
    }

    for (int bi = 0; bi<block_size; bi++) {
      if( (ABS(pc.x-xmin)<EPS || ABS(pc.x-xmax)<EPS ||
           ABS(pc.y-ymin)<EPS || ABS(pc.y-ymax)<EPS
  #ifdef P4_TO_P8
           || ABS(pc.z-zmin)<EPS || ABS(pc.z-zmax)<EPS
           ) && bc[bi].wallType(pc.x,pc.y, pc.z)==DIRICHLET)
  #else
           ) && bc[bi].wallType(pc.x,pc.y)==DIRICHLET)
  #endif
      {
        matrix_has_nullspace = false;
        entry_t ent; ent.n = block_size*global_n_idx + bi; ent.val = 1;
        matrix_entries[block_size*n+bi].push_back(ent);
  #ifdef P4_TO_P8
        rhs_p[block_size*n+bi] = bc[bi].wallValue(pc.x, pc.y, pc.z);
  #else
        rhs_p[block_size*n+bi] = bc[bi].wallValue(pc.x, pc.y);
  #endif

        continue;
      }

      for (int bj = 0; bj<block_size; bj++){
        // compute add
  #ifdef P4_TO_P8
        double add_n = (*add[bi][bj])(pc.x, pc.y, pc.z);
  #else
        double add_n = (*add[bi][bj])(pc.x, pc.y);
  #endif

        if(add_n>EPS) matrix_has_nullspace = false;

        // add the central value
        entry_t ent; ent.n = block_size*global_n_idx + bj; ent.val = volume*add_n;
        matrix_entries[block_size*n + bi].push_back(ent);
      }

      // compute rhs
      if(phi_n<0) {
  #ifdef P4_TO_P8
        rhs_p[block_size*n+bi] = (*this->rhs_m[bi])(pc.x, pc.y, pc.z) * volume;
  #else
        rhs_p[block_size*n+bi] = (*this->rhs_m[bi])(pc.x, pc.y) * volume;
  #endif
      } else {
  #ifdef P4_TO_P8
        rhs_p[block_size*n+bi] = (*this->rhs_p[bi])(pc.x, pc.y, pc.z) * volume;
  #else
        rhs_p[block_size*n+bi] = (*this->rhs_p[bi])(pc.x, pc.y) * volume;
  #endif
      }

    // add contribution for all neighboring points
      for(unsigned int l=0; l<points->size(); ++l) {
#ifdef P4_TO_P8
        double s = (*points)[l].s;
#else
        int k = (l+partition->size()-1) % partition->size();
        double s = ((*partition)[k]-(*partition)[l]).norm_L2();
#endif

        if((*points)[l].n>=0) {
          /* regular point */
#ifdef P4_TO_P8
          Point3 pl = (*points)[l].p;
          double phi_l = interp_phi(pl.x, pl.y, pl.z);
#else
          Point2 pl = (*points)[l].p;
          double phi_l = interp_phi(pl.x, pl.y);
#endif
          double d = (pc - pl).norm_L2();

          for (int bi = 0; bi<block_size; bi++) {
            for (int bj = 0; bj<block_size; bj++) {

#ifdef P4_TO_P8
              if(phi_l<0) mue_l[bi][bj] = (*mu_m[bi][bj])(pl.x, pl.y, pl.z);
              else        mue_l[bi][bj] = (*mu_p[bi][bj])(pl.x, pl.y, pl.z);
#else
              if(phi_l<0) mue_l[bi][bj] = (*mu_m[bi][bj])(pl.x, pl.y);
              else        mue_l[bi][bj] = (*mu_p[bi][bj])(pl.x, pl.y);
#endif
              mue_tmp[bi][bj] = 0.5*(mue_n[bi][bj] + mue_l[bi][bj]);
            }
          }

          inverse((double*)mue_tmp, (double*)mue_inv);
          matmult((double*)mue_n,   (double*)mue_inv, (double*)mue_tmp);
          matmult((double*)mue_tmp, (double*)mue_l,   (double*)mue_h);

          PetscInt global_l_idx;

          if((unsigned int)(*points)[l].n<num_local_voro) {
            global_l_idx = (*points)[l].n + voro_global_offset[p4est->mpirank];
            d_nnz[block_size*n + bi] += block_size;
          } else {
            global_l_idx = voro_ghost_local_num[(*points)[l].n-num_local_voro] + voro_global_offset[voro_ghost_rank[(*points)[l].n-num_local_voro]];
            o_nnz[block_size*n + bi] += block_size;
          }

          for (int bj = 0; bj<block_size; bj++) {

            entry_t ent; ent.n = block_size*global_l_idx + bj; ent.val = -s*mue_h[bi][bj]/d;
            matrix_entries[block_size*n+bi][bj].val += s*mue_h[bi][bj]/d;
            matrix_entries[block_size*n+bi].push_back(ent);

            if(phi_n*phi_l<0) {
#ifdef P4_TO_P8
              Point3 p_ln = (pc+pl)/2;
#else
              Point2 p_ln = (pc+pl)/2;
#endif

              // mue_tmp = mue_h * mue_l^{-1} = mue_n * mue_inv * mue_l * mue_l{-1} = mue_n * mue_inv
#ifdef P4_TO_P8
              rhs_p[block_size*n+bi] += s*mue_h[bi][bj]/d * SIGN(phi_n) * (*u_jump[bj])(p_ln.x, p_ln.y, p_ln.z);
              rhs_p[block_size*n+bi] -= mue_tmp[bi][bj] * s/2 * (*mu_grad_u_jump[bj])(p_ln.x, p_ln.y, p_ln.z);
#else
              rhs_p[block_size*n+bi] += s*mue_h[bi][bj]/d * SIGN(phi_n) * (*u_jump[bj])(p_ln.x, p_ln.y);
              rhs_p[block_size*n+bi] -= mue_tmp[bi][bj] * s/2 * (*mu_grad_u_jump[bj])(p_ln.x, p_ln.y);
#endif
            }

          }

        }
        else /* wall with neumann */
        {
          double x_tmp = pc.x;
          double y_tmp = pc.y;

          /* perturb the corners to differentiate between the edges of the domain ... otherwise 1st order only at the corners */
#ifdef P4_TO_P8
          double z_tmp = pc.z;
          if(pc.x==xmin && ( (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp += 2*EPS;
          if(pc.x==xmax && ( (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp -= 2*EPS;
          if(pc.y==ymin && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp += 2*EPS;
          if(pc.y==ymax && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp -= 2*EPS;
          if(pc.z==zmin && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp += 2*EPS;
          if(pc.z==zmax && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp -= 2*EPS;
          rhs_p[block_size*n+bi] += s*mue_n[bi][bi] * bc[bi].wallValue(x_tmp, y_tmp, z_tmp);
#else
          if(pc.x==xmin && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp += 2*EPS;
          if(pc.x==xmax && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp -= 2*EPS;
          if(pc.y==ymin && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp += 2*EPS;
          if(pc.y==ymax && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp -= 2*EPS;
          rhs_p[block_size*n+bi] += s*mue_n[bi][bi] * bc[bi].wallValue(x_tmp, y_tmp);
#endif
        }
      } // l loop
    } // bi loop
  } // n loop

  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  PetscInt num_owned_global = voro_global_offset[p4est->mpisize];
  PetscInt num_owned_local  = (PetscInt) num_local_voro;

  if (A != NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  /* set up the matrix */
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, block_size*num_owned_local , block_size*num_owned_local,
                     block_size*num_owned_global, block_size*num_owned_global); CHKERRXX(ierr);
  ierr = MatSetBlockSize(A, block_size); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  /* allocate the matrix */
  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  /* fill the matrix with the values */
  for(unsigned int n=0; n<num_local_voro; ++n)
  {
    PetscInt global_n_idx = n+voro_global_offset[p4est->mpirank];
    for (int bi = 0; bi < block_size; bi++) {
      vector<entry_t>& row = matrix_entries[block_size*n+bi];
      for(unsigned int m=0; m<row.size(); ++m) {
        ierr = MatSetValue(A, block_size*global_n_idx + bi, row[m].n, row[m].val, ADD_VALUES); CHKERRXX(ierr);
      }
    }
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
    //ierr = MatSetTransposeNullSpace(A, A_null_space); CHKERRXX(ierr);
//    ierr = MatNullSpaceRemove(A_null_space, rhs, NULL); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);
}

#if 0
void my_p4est_poisson_jump_voronoi_block_t::setup_negative_laplace_rhsvec()
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  for(unsigned int n=0; n<num_local_voro; ++n)
  {
#ifdef P4_TO_P8
    Point3 pc = voro_points[n];
#else
    Point2 pc = voro_points[n];
#endif
    if( (ABS(pc.x-xmin)<EPS || ABS(pc.x-xmax)<EPS ||
         ABS(pc.y-ymin)<EPS || ABS(pc.y-ymax)<EPS
     #ifdef P4_TO_P8
         || ABS(pc.z-zmin)<EPS || ABS(pc.z-zmax)<EPS
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
    const std::vector<Voronoi3DPoint> *points;
#else
    const std::vector<Point2> *partition;
    const std::vector<Voronoi2DPoint> *points;
    voro.get_Partition(partition);
#endif
    voro.get_Points(points);

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
        if(pc.x==xmin && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp += 2*EPS;
        if(pc.x==xmax && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp -= 2*EPS;
        if(pc.y==ymin && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp += 2*EPS;
        if(pc.y==ymax && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp -= 2*EPS;
        if(pc.z==zmin && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp += 2*EPS;
        if(pc.z==zmax && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp -= 2*EPS;
        rhs_p[n] += s*mu_n * bc->wallValue(x_tmp, y_tmp, z_tmp);
#else
        if(pc.x==xmin && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp += 2*EPS;
        if(pc.x==xmax && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp -= 2*EPS;
        if(pc.y==ymin && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp += 2*EPS;
        if(pc.y==ymax && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp -= 2*EPS;
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
#endif


void my_p4est_poisson_jump_voronoi_block_t::interpolate_solution_from_voronoi_to_tree_on_node_n(p4est_locidx_t n,vector<double>& vals) const
{
  PetscErrorCode ierr;

  double *sol_voro_p;
  ierr = VecGetArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);

#ifdef P4_TO_P8
    Point3 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes), node_z_fr_n(n, p4est, nodes));
#else
    Point2 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes));
#endif

#ifdef P4_TO_P8
    Point3 pm;
#else
    Point2 pm;
#endif
    /* first check if the node is a voronoi point */
    for(unsigned int m=0; m<grid2voro[n].size(); ++m)
    {
      pm = voro_points[grid2voro[n][m]];
      if((pn-pm).norm_L2()<EPS)
      {
        for (int i=0; i<block_size; i++) {
          vals[i] = sol_voro_p[block_size*grid2voro[n][m] + i];
        }
        ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);
        return;
      }
    }

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    /* if not a grid point, gather all the neighbor voro points and find the
     * three closest with the same sign for phi */
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;

    std::vector<p4est_locidx_t> ngbd_quads;

#ifdef P4_TO_P8
    std::vector<p4est_quadrant_t> tmp;
    for(char i=-1; i<=1; i+=2)
    {
      for(char j=-1; j<=1; j+=2)
      {
        for(char k=-1; k<=1; k+=2)
        {
          ngbd_n->find_neighbor_cell_of_node(n,  i,  j, k, quad_idx, tree_idx);
          if(quad_idx>=0)
          {
            ngbd_quads.push_back(quad_idx);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, 0, 0);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, 0, j, 0);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, 0, 0, k);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, j, 0);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, 0, k);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, 0, j, k);
            ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, j, k);
            for(unsigned int m=0; m<tmp.size(); ++m)
              ngbd_quads.push_back(tmp[m].p.piggy3.local_num);
            tmp.clear();
          }
        }
      }
    }
#else
    std::vector<p4est_quadrant_t> tmp;
    for(char i=-1; i<=1; i+=2)
    {
      for(char j=-1; j<=1; j+=2)
      {
        ngbd_n->find_neighbor_cell_of_node(n,  i,  j, quad_idx, tree_idx);
        if(quad_idx>=0)
        {
          ngbd_quads.push_back(quad_idx);
          ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, 0);
          ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, 0, j);
          ngbd_c->find_neighbor_cells_of_cell(tmp, quad_idx, tree_idx, i, j);
          for(unsigned int m=0; m<tmp.size(); ++m)
            ngbd_quads.push_back(tmp[m].p.piggy3.local_num);
          tmp.clear();
        }
      }
    }
#endif

    /* now find the two voronoi points closest to the node */
    double phi_n = phi_p[n];
#ifdef P4_TO_P8
    double di[] = {DBL_MAX, DBL_MAX, DBL_MAX, DBL_MAX};
    unsigned int ni[] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
#else
    double di[] = {DBL_MAX, DBL_MAX, DBL_MAX};
    unsigned int ni[] = {UINT_MAX, UINT_MAX, UINT_MAX};
#endif
    for(unsigned int k=0; k<ngbd_quads.size(); ++k)
    {
      for(int dir=0; dir<P4EST_CHILDREN; ++dir)
      {
        p4est_locidx_t n_idx = nodes->local_nodes[P4EST_CHILDREN*ngbd_quads[k] + dir];
        for(unsigned int m=0; m<grid2voro[n_idx].size(); ++m)
        {
          /* if point is not already in the list */
          if( ni[0]!=grid2voro[n_idx][m] &&
              ni[1]!=grid2voro[n_idx][m] )
          {
            pm = voro_points[grid2voro[n_idx][m]];
#ifdef P4_TO_P8
            double xyz[] = {pm.x, pm.y, pm.z};
#else
            double xyz[] = {pm.x, pm.y};
#endif
            p4est_quadrant_t quad;
            std::vector<p4est_quadrant_t> remote_matches;

            int rank = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);
            if(rank!=-1)
            {
#ifdef P4_TO_P8
              double phi_m = interp_phi(pm.x, pm.y, pm.z);
#else
              double phi_m = interp_phi(pm.x, pm.y);
#endif
              if(phi_m*phi_n>=0)
              {
                double d = (pm-pn).norm_L2();

                if(d<di[0])
                {
                  ni[1]=ni[0]; di[1]=di[0];
                  ni[0]=grid2voro[n_idx][m]; di[0]=d;
                }
                else if(d<di[1])
                {
                  ni[1]=grid2voro[n_idx][m]; di[1]=d;
                }
              }
            } else {
              throw std::runtime_error("Found rank = -1 in voronoi interpolaiton. This should not happen. SHIT SHIT SHIT");
            }
          }
        }
      }
    }

    assert(ni[0] != UINT_MAX && ni[1] != UINT_MAX);
#ifdef P4_TO_P8
    Point3 p0(voro_points[ni[0]]);
    Point3 p1(voro_points[ni[1]]);
#else
    Point2 p0(voro_points[ni[0]]);
    Point2 p1(voro_points[ni[1]]);
#endif

    /* find a third point forming a non-flat triangle */
    for(unsigned int k=0; k<ngbd_quads.size(); ++k)
    {
      for(int dir=0; dir<P4EST_CHILDREN; ++dir)
      {
        p4est_locidx_t n_idx = nodes->local_nodes[P4EST_CHILDREN*ngbd_quads[k] + dir];
        for(unsigned int m=0; m<grid2voro[n_idx].size(); ++m)
        {
          /* if point is not already in the list */
          if( ni[0]!=grid2voro[n_idx][m] &&
              ni[1]!=grid2voro[n_idx][m] &&
              ni[2]!=grid2voro[n_idx][m])
          {
            pm = voro_points[grid2voro[n_idx][m]];
#ifdef P4_TO_P8
            double xyz[] = {pm.x, pm.y, pm.z};
#else
            double xyz[] = {pm.x, pm.y};
#endif
            p4est_quadrant_t quad;
            std::vector<p4est_quadrant_t> remote_matches;

            int rank = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);
            if(rank!=-1)
            {
#ifdef P4_TO_P8
              double phi_m = interp_phi(pm.x, pm.y, pm.z);
#else
              double phi_m = interp_phi(pm.x, pm.y);
#endif
              if(phi_m*phi_n>=0)
              {
                double d = (pm-pn).norm_L2();

#ifdef P4_TO_P8
                if( d<di[2] && ((p0-p1).normalize().cross((pm-p1).normalize())).norm_L2() > sin(PI/10) )
#else
                if(d<di[2] && ABS((p0-p1).normalize().cross((pm-p1).normalize())) > sin(PI/5))
#endif
                {
                  ni[2]=grid2voro[n_idx][m]; di[2]=d;
                }
              }
            } else {
              throw std::runtime_error("Found rank = -1 in voronoi interpolaiton. This should not happen. SHIT SHIT SHIT");
            }
          }
        }
      }
    }

    assert(ni[2] != UINT_MAX);
#ifdef P4_TO_P8
    Point3 p2(voro_points[ni[2]]);
#else
    Point2 p2(voro_points[ni[2]]);
#endif

#ifdef P4_TO_P8
    /* in 3D, found a fourth point to form a non-flat tetrahedron */
    for(unsigned int k=0; k<ngbd_quads.size(); ++k)
    {
      for(int dir=0; dir<P4EST_CHILDREN; ++dir)
      {
        p4est_locidx_t n_idx = nodes->local_nodes[P4EST_CHILDREN*ngbd_quads[k] + dir];
        for(unsigned int m=0; m<grid2voro[n_idx].size(); ++m)
        {
          /* if point is not already in the list */
          if( ni[0]!=grid2voro[n_idx][m] &&
              ni[1]!=grid2voro[n_idx][m] &&
              ni[2]!=grid2voro[n_idx][m] &&
              ni[3]!=grid2voro[n_idx][m])
          {
            pm = voro_points[grid2voro[n_idx][m]];
            double xyz[] = {pm.x, pm.y, pm.z};
            p4est_quadrant_t quad;
            std::vector<p4est_quadrant_t> remote_matches;

            int rank = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);
            if(rank!=-1)
            {
              double phi_m = interp_phi(pm.x, pm.y, pm.z);
              if(phi_m*phi_n>=0)
              {
                double d = (pm-pn).norm_L2();

                Point3 n = (p1-p0).cross(p2-p0).normalize();
                double h = ABS((pm-p0).dot(n));

                if(d<di[3] && h > diag_min/5)
                {
                  ni[3]=grid2voro[n_idx][m]; di[3]=d;
                }
              }
            } else {
              throw std::runtime_error("Found rank = -1 in voronoi interpolaiton. This should not happen. SHIT SHIT SHIT");
            }
          }
        }
      }
    }

    assert(ni[3] != UINT_MAX);
    Point3 p3(voro_points[ni[3]]);
#endif

    /* make sure we found 3 points */
#ifdef P4_TO_P8
    if(di[0]==DBL_MAX || di[1]==DBL_MAX || di[2]==DBL_MAX || di[3]==DBL_MAX)
#else
    if(di[0]==DBL_MAX || di[1]==DBL_MAX || di[2]==DBL_MAX)
#endif
    {
      std::cerr << "my_p4est_poisson_jump_nodes_voronoi_t->interpolate_solution_from_voronoi_to_tree: not enough points found." << std::endl;
      for (int i=0; i<block_size; i++) {
        vals[i] = sol_voro_p[block_size*ni[0] + i];
      }
      ierr = VecRestoreArray(phi     , &phi_p     ); CHKERRXX(ierr);
      ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);
      return;
    }

#ifdef P4_TO_P8
    if(ni[0]==ni[1] || ni[0]==ni[2] || ni[0]==ni[3] || ni[1]==ni[2] || ni[1]==ni[3] || ni[2]==ni[3])
#else
    if(ni[0]==ni[1] || ni[0]==ni[2] || ni[1]==ni[2])
#endif
      std::cerr << "my_p4est_poisson_jump_nodes_voronoi_t->interpolate_solution_from_voronoi_to_tree: point is double !" << std::endl;

    vector<double> f0(block_size), f1(block_size), f2(block_size);
#ifdef P4_TO_P8
    vector<double> f3(block_size);
#endif
    for (int i=0; i<block_size; i++) {
      f0[i] = sol_voro_p[block_size*ni[0]+i];
      f1[i] = sol_voro_p[block_size*ni[1]+i];
      f2[i] = sol_voro_p[block_size*ni[2]+i];
#ifdef P4_TO_P8
      f3[i] = sol_voro_p[block_size*ni[3]+i];
#endif
    }

#ifdef P4_TO_P8
    double det = ( -( p1.x*p2.y*p3.z + p2.x*p3.y*p1.z + p3.x*p1.y*p2.z - p3.x*p2.y*p1.z - p2.x*p1.y*p3.z - p1.x*p3.y*p2.z )
                   +( p0.x*p2.y*p3.z + p2.x*p3.y*p0.z + p3.x*p0.y*p2.z - p3.x*p2.y*p0.z - p2.x*p0.y*p3.z - p0.x*p3.y*p2.z )
                   -( p0.x*p1.y*p3.z + p1.x*p3.y*p0.z + p3.x*p0.y*p1.z - p3.x*p1.y*p0.z - p1.x*p0.y*p3.z - p0.x*p3.y*p1.z )
                   +( p0.x*p1.y*p2.z + p1.x*p2.y*p0.z + p2.x*p0.y*p1.z - p2.x*p1.y*p0.z - p1.x*p0.y*p2.z - p0.x*p2.y*p1.z ) );
#else
    double det = p0.x*p1.y + p1.x*p2.y + p2.x*p0.y - p1.x*p0.y - p2.x*p1.y - p0.x*p2.y;
#endif

#ifdef CASL_THROWS
    if(ABS(det)<EPS)
    {
      std::cout << p0 << p1 << p2
#ifdef P4_TO_P8
                << p3;
#else
                   ;
#endif
      throw std::invalid_argument("[CASL_ERROR]: interpolation_Voronoi: could not invert system ...");
    }
#endif

    ierr = VecRestoreArray(phi     , &phi_p     ); CHKERRXX(ierr);
    ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);

#ifdef P4_TO_P8

    /*
     * solving A*C = F,
     *     | x0 y0 z0 1 |      | c0 |      | f0 |
     *     | x1 y1 z1 1 |      | c1 |      | f1 |
     * A = | x2 y2 z2 1 |, C = | c2 |, F = | f2 |
     *     | x3 y3 z3 1 |      | c3 |      | f3 |
     *
     *               | b00 b01 b02 b03 |       | c0 |        | f0 |
     *  -1           | b10 b11 b12 b13 |       | c1 |    -1  | f1 |
     * A   = 1/det * | b20 b21 b22 b23 |, and  | c2 | = A  * | f2 |
     *               | b30 b31 b32 b33 |       | c3 |        | f3 |
     *
     */

    double b00 =  ( p1.y*p2.z + p2.y*p3.z + p3.y*p1.z - p1.y*p3.z - p2.y*p1.z - p3.y*p2.z );
    double b01 = -( p0.y*p2.z + p2.y*p3.z + p3.y*p0.z - p0.y*p3.z - p2.y*p0.z - p3.y*p2.z );
    double b02 =  ( p0.y*p1.z + p1.y*p3.z + p3.y*p0.z - p0.y*p3.z - p1.y*p0.z - p3.y*p1.z );
    double b03 = -( p0.y*p1.z + p1.y*p2.z + p2.y*p0.z - p0.y*p2.z - p1.y*p0.z - p2.y*p1.z );

    double b10 = -( p1.x*p2.z + p2.x*p3.z + p3.x*p1.z - p1.x*p3.z - p2.x*p1.z - p3.x*p2.z );
    double b11 =  ( p0.x*p2.z + p2.x*p3.z + p3.x*p0.z - p0.x*p3.z - p2.x*p0.z - p3.x*p2.z );
    double b12 = -( p0.x*p1.z + p1.x*p3.z + p3.x*p0.z - p0.x*p3.z - p1.x*p0.z - p3.x*p1.z );
    double b13 =  ( p0.x*p1.z + p1.x*p2.z + p2.x*p0.z - p0.x*p2.z - p1.x*p0.z - p2.x*p1.z );

    double b20 =  ( p1.x*p2.y + p2.x*p3.y + p3.x*p1.y - p1.x*p3.y - p2.x*p1.y - p3.x*p2.y );
    double b21 = -( p0.x*p2.y + p2.x*p3.y + p3.x*p0.y - p0.x*p3.y - p2.x*p0.y - p3.x*p2.y );
    double b22 =  ( p0.x*p1.y + p1.x*p3.y + p3.x*p0.y - p0.x*p3.y - p1.x*p0.y - p3.x*p1.y );
    double b23 = -( p0.x*p1.y + p1.x*p2.y + p2.x*p0.y - p0.x*p2.y - p1.x*p0.y - p2.x*p1.y );

    double b30 = -( p1.x*p2.y*p3.z + p2.x*p3.y*p1.z + p3.x*p1.y*p2.z - p1.x*p3.y*p2.z - p2.x*p1.y*p3.z - p3.x*p2.y*p1.z );
    double b31 =  ( p0.x*p2.y*p3.z + p2.x*p3.y*p0.z + p3.x*p0.y*p2.z - p0.x*p3.y*p2.z - p2.x*p0.y*p3.z - p3.x*p2.y*p0.z );
    double b32 = -( p0.x*p1.y*p3.z + p1.x*p3.y*p0.z + p3.x*p0.y*p1.z - p0.x*p3.y*p1.z - p1.x*p0.y*p3.z - p3.x*p1.y*p0.z );
    double b33 =  ( p0.x*p1.y*p2.z + p1.x*p2.y*p0.z + p2.x*p0.y*p1.z - p0.x*p2.y*p1.z - p1.x*p0.y*p2.z - p2.x*p1.y*p0.z );

    double c0, c1, c2, c3;
    for (int i=0; i<block_size; i++) {
      c0 = (b00*f0[i] + b01*f1[i] + b02*f2[i] + b03*f3[i]) / det;
      c1 = (b10*f0[i] + b11*f1[i] + b12*f2[i] + b13*f3[i]) / det;
      c2 = (b20*f0[i] + b21*f1[i] + b22*f2[i] + b23*f3[i]) / det;
      c3 = (b30*f0[i] + b31*f1[i] + b32*f2[i] + b33*f3[i]) / det;

      vals[i] = c0*pn.x + c1*pn.y + c2*pn.z + c3;
    }


#else

    double c0, c1, c2;
    for (int i=0; i<block_size; i++) {
      c0 = ( (p1.y* 1- 1*p2.y)*f0[i] + ( 1*p2.y-p0.y* 1)*f1[i] + (p0.y* 1- 1*p1.y)*f2[i] ) / det;
      c1 = ( ( 1*p2.x-p1.x* 1)*f0[i] + (p0.x* 1- 1*p2.x)*f1[i] + ( 1*p1.x-p0.x* 1)*f2[i] ) / det;
      c2 = ( (p1.x*p2.y-p2.x*p1.y)*f0[i] + (p2.x*p0.y-p0.x*p2.y)*f1[i] + (p0.x*p1.y-p1.x*p0.y)*f2[i] ) / det;

      vals[i] = c0*pn.x + c1*pn.y + c2;
    }
#endif
}



void my_p4est_poisson_jump_voronoi_block_t::interpolate_solution_from_voronoi_to_tree(Vec solution[]) const
{
  PetscErrorCode ierr;

  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_interpolate_to_tree, phi, sol_voro, solution, 0); CHKERRXX(ierr);

  double* solution_p[block_size];
  for (int i=0; i<block_size; i++){
    ierr = VecGetArray(solution[i], &solution_p[i]); CHKERRXX(ierr);
  }

  /* for debugging, compute the error on the voronoi mesh */
  // bousouf
  if(0)
  {
    double *sol_voro_p;
    ierr = VecGetArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);

    double err = 0;
    for(unsigned int n=0; n<num_local_voro; ++n)
    {
      double u_ex;
#ifdef P4_TO_P8
      Point3 pc = voro_points[n];

      double phi_n = interp_phi(pc.x, pc.y, pc.z);
      u_ex = cos(pc.x)*sin(pc.y)*exp(pc.z);
      if(phi_n<0) u_ex = exp(pc.z);
      else        u_ex = cos(pc.x)*sin(pc.y);
#else
      Point2 pc = voro_points[n];

      double phi_n = interp_phi(pc.x, pc.y);
//      u_ex = cos(pc.x)*sin(pc.y);
      if(phi_n<0) u_ex = exp(pc.x);
      else        u_ex = cos(pc.x)*sin(pc.y);
#endif

      err = std::max(err, ABS(u_ex - sol_voro_p[n]));
    }

    ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);

    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
//    PetscPrintf(p4est->mpicomm, "Error on voronoi : %g\n", err);
  }

  vector<double> vals(block_size);
  for(size_t i=0; i<ngbd_n->get_layer_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_layer_node(i);
    interpolate_solution_from_voronoi_to_tree_on_node_n(n, vals);
    for (int s=0; s<block_size; s++) {
      solution_p[s][n] = vals[s];
    }
  }

  for (int i = 0; i<block_size; i++) {
    ierr = VecGhostUpdateBegin(solution[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for(size_t i=0; i<ngbd_n->get_local_size(); ++i)
  {
    p4est_locidx_t n = ngbd_n->get_local_node(i);
    interpolate_solution_from_voronoi_to_tree_on_node_n(n, vals);
    for (int s=0; s<block_size; s++) {
      solution_p[s][n] = vals[s];
    }
  }

  for (int i = 0; i<block_size; i++) {
    ierr = VecGhostUpdateEnd  (solution[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  }

  for (int i = 0; i<block_size; i++) {
    ierr = VecRestoreArray(solution[i], &solution_p[i]); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_interpolate_to_tree, phi, sol_voro, solution, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_jump_voronoi_block_t::write_stats(const char *path) const
{
  std::vector<unsigned int> nodes_voro(p4est->mpisize, 0);
  std::vector<unsigned int> indep_voro(p4est->mpisize, 0);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    Point3 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes), node_z_fr_n(n, p4est, nodes));
#else
    Point2 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes));
#endif

    /* first check if the node is a voronoi point */
    for(unsigned int m=0; m<grid2voro[n].size(); ++m)
    {
      if((pn-voro_points[grid2voro[n][m]]).norm_L2()<EPS)
      {
        nodes_voro[p4est->mpirank]++;
        goto next_point;
      }
    }

    indep_voro[p4est->mpirank]++;
next_point:
    ;
  }

  MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED, &nodes_voro[0], 1, MPI_UNSIGNED, p4est->mpicomm);
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_UNSIGNED, &indep_voro[0], 1, MPI_UNSIGNED, p4est->mpicomm);

  /* write voronoi stats */
  if(p4est->mpirank==0)
  {
    FILE *f = fopen(path, "w");
    fprintf(f, "%% rank  |  nb_voro  |  nb_indep_voro  |  nb_nodes_voro\n");
    for(int i=0; i<p4est->mpisize; ++i)
      fprintf(f, "%d %d %u %u\n", i, voro_global_offset[i+1]-voro_global_offset[i], indep_voro[i], nodes_voro[i]);
    fclose(f);
  }
}




void my_p4est_poisson_jump_voronoi_block_t::print_voronoi_VTK(const char* path) const
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
//  Voronoi3D::print_VTK_Format(voro, name, xmin, xmax, ymin, ymax, zmin, zmax, false, false, false);
#else
  Voronoi2D::print_VTK_Format(voro, name);
#endif
}


void my_p4est_poisson_jump_voronoi_block_t::check_voronoi_partition() const
{
  PetscErrorCode ierr;
  ierr = PetscPrintf(p4est->mpicomm, "Checking partition ...\n"); CHKERRXX(ierr);
#ifdef P4_TO_P8
  std::vector<Voronoi3D> voro(num_local_voro);
  const std::vector<Voronoi3DPoint> *points;
  const std::vector<Voronoi3DPoint> *pts;
#else
  std::vector<Voronoi2D> voro(num_local_voro);
  const std::vector<Voronoi2DPoint> *points;
  const std::vector<Voronoi2DPoint> *pts;
#endif

  std::vector< std::vector<check_comm_t> > buff_send(p4est->mpisize);

  for(unsigned int n=0; n<num_local_voro; ++n)
    compute_voronoi_cell(n, voro[n]);

  int nb_bad = 0;
  for(unsigned int n=0; n<num_local_voro; ++n)
  {
    voro[n].get_Points(points);

    for(unsigned int m=0; m<points->size(); ++m)
    {
      if((*points)[m].n>=0)
      {
        if((*points)[m].n<(int) num_local_voro)
        {
          voro[(*points)[m].n].get_Points(pts);
          bool ok = false;
          for(unsigned int k=0; k<pts->size(); ++k)
          {
            if((*pts)[k].n==(int) n)
            {
              ok = true;
              continue;
            }
          }

          if(ok==false)
          {
            std::cout << p4est->mpirank << " found bad voronoi cell for point # " << n << " : " << (*points)[m].n << ", \t Centerd on : " << voro[n].get_Center_Point();
            nb_bad++;
          }
        }
        else if(voro_ghost_rank[(*points)[m].n-num_local_voro]<p4est->mpirank)
        {
          check_comm_t tmp;
          tmp.n = n;
          tmp.k = voro_ghost_local_num[(*points)[m].n-num_local_voro];
          buff_send[voro_ghost_rank[(*points)[m].n-num_local_voro]].push_back(tmp);
        }
      }
    }
  }

  /* initiate communication */
  std::vector<MPI_Request> req(p4est->mpirank);
  for(int r=0; r<p4est->mpirank; ++r)
  {
    MPI_Isend(&buff_send[r][0], buff_send[r].size()*sizeof(check_comm_t), MPI_BYTE, r, 8, p4est->mpicomm, &req[r]);
  }

  /* now receive */
  int nb_recv = p4est->mpisize-(p4est->mpirank+1);
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

        voro[local_idx].get_Points(pts);
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
          std::cout << p4est->mpirank << " found bad ghost voronoi cell for point # " << local_idx << " : " << ghost_idx << ", \t Centerd on : " << voro[local_idx].get_Center_Point();
          nb_bad++;
        }
    }

    nb_recv--;
  }

  MPI_Waitall(req.size(), &req[0], MPI_STATUSES_IGNORE);

  MPI_Allreduce(MPI_IN_PLACE, (void*) &nb_bad, 1, MPI_INT, MPI_SUM, p4est->mpicomm);

  if(nb_bad==0) { ierr = PetscPrintf(p4est->mpicomm, "Partition is good.\n"); CHKERRXX(ierr); }
  else          { ierr = PetscPrintf(p4est->mpicomm, "Partition is NOT good, %d problem found.\n", nb_bad); CHKERRXX(ierr); }
}
