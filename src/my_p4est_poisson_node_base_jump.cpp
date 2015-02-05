#include <src/my_p4est_poisson_node_base_jump.h>
#include <src/my_p4est_refine_coarsen.h>

#include <algorithm>

#include <src/petsc_compatibility.h>
#include <src/CASL_math.h>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_PoissonSolverNodeBasedJump_matrix_preallocation;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_matrix_setup;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_rhsvec_setup;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_KSPSolve;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_solve;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_compute_voronoi_mesh;
extern PetscLogEvent log_PoissonSolverNodeBasedJump_interpolate_to_tree;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bc_strength 1.0

PoissonSolverNodeBaseJump::PoissonSolverNodeBaseJump(const my_p4est_node_neighbors_t *node_neighbors,
                                                     const my_p4est_cell_neighbors_t *cell_neighbors)
  : ngbd_n(node_neighbors),  ngbd_c(cell_neighbors), myb(node_neighbors->myb),
    p4est(node_neighbors->p4est), ghost(node_neighbors->ghost), nodes(node_neighbors->nodes),
    phi(NULL), rhs(NULL), sol_voro(NULL),
    global_voro_offset(p4est->mpisize),
    interp_phi(NULL, *node_neighbors, linear),
    rhs_m(NULL, *node_neighbors, linear),
    rhs_p(NULL, *node_neighbors, linear),
    local_mu(false), local_add(false),
    local_u_jump(false), local_mu_grad_u_jump(false),
    mu_m(&mu_constant), mu_p(&mu_constant), add(&add_constant),
    u_jump(&zero), mu_grad_u_jump(&zero),
    A(PETSC_NULL), A_null_space(PETSC_NULL), ksp(PETSC_NULL),
    is_voronoi_partition_constructed(false), is_matrix_computed(false), matrix_has_nullspace(false)
{
  // set up the KSP solver
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  // compute grid parameters
  // NOTE: Assuming all trees are of the same size [0, 1]^d
  xmin = 0; xmax = myb->nxyztrees[0];
  ymin = 0; ymax = myb->nxyztrees[1];

  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin_ = p4est->connectivity->vertices[3*vm + 0];
  double ymin_ = p4est->connectivity->vertices[3*vm + 1];
  double xmax_ = p4est->connectivity->vertices[3*vp + 0];
  double ymax_ = p4est->connectivity->vertices[3*vp + 1];
  dx_min = (xmax_-xmin_) / pow(2.,(double) data->max_lvl);
  dy_min = (ymax_-ymin_) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  zmin = 0; zmax = myb->nxyztrees[2];
  double zmin_ = p4est->connectivity->vertices[3*vm + 2];
  double zmax_ = p4est->connectivity->vertices[3*vp + 2];
  dz_min = (zmax_-zmin_) / pow(2.,(double) data->max_lvl);
#endif
#ifdef P4_TO_P8
  d_min = MIN(dx_min, dy_min, dz_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min + dz_min*dz_min);
#else
  d_min = MIN(dx_min, dy_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min);
#endif
}


PoissonSolverNodeBaseJump::~PoissonSolverNodeBaseJump()
{
  if(A            != PETSC_NULL) { ierr = MatDestroy(A);                     CHKERRXX(ierr); }
  if(A_null_space != PETSC_NULL) { ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr); }
  if(ksp          != PETSC_NULL) { ierr = KSPDestroy(ksp);                   CHKERRXX(ierr); }
  if(rhs          != PETSC_NULL) { ierr = VecDestroy(rhs);                   CHKERRXX(ierr); }
  if(local_mu)             { delete mu_m; delete mu_p; }
  if(local_add)            { delete add; }
  if(local_u_jump)         { delete u_jump; }
  if(local_mu_grad_u_jump) { delete mu_grad_u_jump; }
}


PetscErrorCode PoissonSolverNodeBaseJump::VecCreateGhostVoronoiRhs()
{
  PetscErrorCode ierr = 0;
  PetscInt num_local = num_local_voro;
  PetscInt num_global = global_voro_offset[p4est->mpisize];

  std::vector<PetscInt> ghost_voro(voro.size() - num_local, 0);

  for (size_t i = 0; i<ghost_voro.size(); ++i)
  {
    ghost_voro[i] = voro_ghost_local_num[i] + global_voro_offset[voro_ghost_rank[i]];
  }

  if(rhs!=PETSC_NULL) VecDestroy(rhs);

  ierr = VecCreateGhost(p4est->mpicomm, num_local_voro, num_global,
                        ghost_voro.size(), (const PetscInt*)&ghost_voro[0], &rhs); CHKERRQ(ierr);
  ierr = VecSetFromOptions(rhs); CHKERRQ(ierr);

  return ierr;
}


void PoissonSolverNodeBaseJump::set_phi(Vec phi)
{
  this->phi = phi;
  interp_phi.set_input(phi);
}


void PoissonSolverNodeBaseJump::set_rhs(Vec rhs_m, Vec rhs_p)
{
  this->rhs_m.set_input(rhs_m);
  this->rhs_p.set_input(rhs_p);
}


void PoissonSolverNodeBaseJump::set_diagonal(double add)
{
  if(local_add) { delete this->add; local_add = false; }
  add_constant.set(add);
  this->add = &add_constant;
}

void PoissonSolverNodeBaseJump::set_diagonal(Vec add)
{
  if(local_add) delete this->add;
  this->add = new InterpolatingFunctionNodeBaseHost(add, *ngbd_n, linear);
  local_add = true;
}


void PoissonSolverNodeBaseJump::set_bc(BoundaryConditions2D& bc)
{
  this->bc = &bc;
  is_matrix_computed = false;
}


void PoissonSolverNodeBaseJump::set_mu(double mu)
{
  if(local_mu) { delete mu_m; delete mu_p; local_mu = false; }
  mu_constant.set(mu);
  mu_m = &mu_constant;
  mu_p = &mu_constant;
}


void PoissonSolverNodeBaseJump::set_mu(Vec mu_m, Vec mu_p)
{
  if(local_mu) { delete this->mu_m; delete this->mu_p; }
  this->mu_m = new InterpolatingFunctionNodeBaseHost(mu_m, *ngbd_n, linear);
  this->mu_p = new InterpolatingFunctionNodeBaseHost(mu_p, *ngbd_n, linear);
  local_mu = true;
}


void PoissonSolverNodeBaseJump::set_u_jump(Vec u_jump)
{
  if(local_u_jump) delete this->u_jump;
  this->u_jump = new InterpolatingFunctionNodeBaseHost(u_jump, *ngbd_n, linear);
  local_u_jump = true;
}

void PoissonSolverNodeBaseJump::set_mu_grad_u_jump(Vec mu_grad_u_jump)
{
  if(local_mu_grad_u_jump) delete this->mu_grad_u_jump;
  this->mu_grad_u_jump = new InterpolatingFunctionNodeBaseHost(mu_grad_u_jump, *ngbd_n, linear);
  local_mu_grad_u_jump = true;
}


void PoissonSolverNodeBaseJump::preallocate_matrix()
{
  /* enable logging for the preallocation */
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = global_voro_offset[p4est->mpisize];
  PetscInt num_owned_local  = (PetscInt) num_local_voro;

  if (A != NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  /* set up the matrix */
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  std::vector<PetscInt> d_nnz(num_local_voro, 1), o_nnz(num_local_voro, 0);

  for(unsigned int n=0; n<num_local_voro; ++n)
  {
    Point2 pc = voro[n].get_Center_Point();
    if( (ABS(pc.x-xmin)<EPS || ABS(pc.x-xmax)<EPS ||
         ABS(pc.y-ymin)<EPS || ABS(pc.y-ymax)<EPS) &&
        bc->wallType(pc.x,pc.y)==DIRICHLET)
      continue;

    const std::vector<Point2> *partition;
    const std::vector<Voronoi2DPoint> *points;
    voro[n].get_Partition(points, partition);

    for(unsigned int m=0; m<points->size(); ++m)
    {
      if((*points)[m].n>=0)
      {
        if((unsigned int) (*points)[m].n<num_local_voro) d_nnz[n]++;
        else                                             o_nnz[n]++;
      }
    }
  }

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void PoissonSolverNodeBaseJump::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_solve, A, rhs, ksp, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  if(bc == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");

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

  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);  
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  /* first compute the voronoi partition */
  if(!is_voronoi_partition_constructed)
  {
    is_voronoi_partition_constructed = true;
    compute_voronoi_mesh();
  }

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  if(!is_matrix_computed)
  {
    matrix_has_nullspace = true;
    setup_negative_laplace_matrix();
    is_matrix_computed = true;
    ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  } else {
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
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  /* setup rhs */
  setup_negative_laplace_rhsvec();

  /* set the nullspace */
  if (matrix_has_nullspace)
    ierr = KSPSetNullSpace(ksp, A_null_space); CHKERRXX(ierr);

  /* Solve the system */
  ierr = VecDuplicate(rhs, &sol_voro); CHKERRXX(ierr);
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_KSPSolve, ksp, rhs, sol_voro, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, sol_voro); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_PoissonSolverNodeBasedJump_KSPSolve, ksp, rhs, sol_voro, 0); CHKERRXX(ierr);

  /* update ghosts */
  ierr = VecGhostUpdateBegin(sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  /* interpolate the solution back onto the original mesh */
  interpolate_solution_from_voronoi_to_tree(solution);

  ierr = VecDestroy(sol_voro); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
}


void PoissonSolverNodeBaseJump::compute_voronoi_mesh()
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_compute_voronoi_mesh, 0, 0, 0, 0); CHKERRXX(ierr);

  if(grid2voro.size()!=0)
  {
    for(unsigned int n=0; n<grid2voro.size(); ++n)
      grid2voro[n].clear();
  }
  grid2voro.resize(nodes->indep_nodes.elem_count);

  voro.clear();

  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  /* find the projected points associated to shared nodes
   * if a projected point is shared, all larger rank are informed.
   * The goal here is to avoid building two close projected points at a processor boundary
   * and to have a consistent partition across processes
   */
  std::vector< std::vector<added_point_t> > buff_shared_added_points_send(p4est->mpisize);
  std::vector< std::vector<added_point_t> > buff_shared_added_points_recv(p4est->mpisize);
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    size_t num_sharers = (size_t) node->pad8;
    if(num_sharers>0)
    {
      double p_00, p_m0, p_p0, p_0m, p_0p;
      (*ngbd_n)[n].ngbd_with_quadratic_interpolation(phi_p, p_00, p_m0, p_p0, p_0m, p_0p);

      if(p_00*p_m0<=0 || p_00*p_p0<=0 || p_00*p_0m<=0 || p_00*p_0p<=0)
      {
        double d = phi_p[n];
        Point2 dp((*ngbd_n)[n].dx_central(phi_p), (*ngbd_n)[n].dy_central(phi_p));
        dp /= dp.norm_L2();
        double xn = node_x_fr_n(n, p4est, nodes);
        double yn = node_y_fr_n(n, p4est, nodes);
        added_point_t added_point_n;
        added_point_n.x = xn-d*dp.x;
        added_point_n.y = yn-d*dp.y;
        added_point_n.dx = dp.x;
        added_point_n.dy = dp.y;

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
          buff_shared_added_points_send[sharers[s]].push_back(added_point_n);
        }
        buff_shared_added_points_recv[p4est->mpirank].push_back(added_point_n);
      }
    }
  }

  /* NOTE: not true anymore .. this can be improved */
  /* send the shared points to the corresponding neighbors ranks
   * note that some messages have a size 0 since the processes can't know who is going to send them data
   * in order to find that out, one needs to call ngbd_with_quadratic_interpolation on ghost nodes...
   */
  for(int r=0; r<p4est->mpisize; ++r)
  {
    if(r!=p4est->mpirank)
    {
      MPI_Request req;
      MPI_Isend(&buff_shared_added_points_send[r][0], buff_shared_added_points_send[r].size()*sizeof(added_point_t), MPI_BYTE, r, 4, p4est->mpicomm, &req);
      MPI_Request_free(&req);
    }
  }

  /* now receive the points */
  int nb_rcv = p4est->mpisize-1;
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
  double band = diag_min/5;
  std::vector<Point2> added_points;
  std::vector<Point2> added_points_grad;
  for(int r=0; r<p4est->mpisize; ++r)
  {
    for(unsigned int m=0; m<buff_shared_added_points_recv[r].size(); ++m)
    {
      Point2 p(buff_shared_added_points_recv[r][m].x, buff_shared_added_points_recv[r][m].y);

      bool already_added = false;
      for(unsigned int k=0; k<added_points.size(); ++k)
      {
        if((p-added_points[k]).norm_L2() < diag_min/5)
        {
          already_added = true;
          break;
        }
      }

      if(!already_added)
      {
        added_points.push_back(p);
        Point2 dp(buff_shared_added_points_recv[r][m].dx, buff_shared_added_points_recv[r][m].dy);
        added_points_grad.push_back(dp);
      }
    }

    buff_shared_added_points_send[r].clear();
    buff_shared_added_points_recv[r].clear();
  }

  buff_shared_added_points_send.clear();
  buff_shared_added_points_recv.clear();

  /* add the local points to the list of projected points */
  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double p_00, p_m0, p_p0, p_0m, p_0p;
    (*ngbd_n)[n].ngbd_with_quadratic_interpolation(phi_p, p_00, p_m0, p_p0, p_0m, p_0p);
    double xn = node_x_fr_n(n, p4est, nodes);
    double yn = node_y_fr_n(n, p4est, nodes);

    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    size_t num_sharers = (size_t) node->pad8;

    bool is_interface = (p_00*p_m0<=0 || p_00*p_p0<=0 || p_00*p_0m<=0 || p_00*p_0p<=0);

    if(is_interface && num_sharers==0)
    {
      double d = phi_p[n];
      Point2 dp((*ngbd_n)[n].dx_central(phi_p), (*ngbd_n)[n].dy_central(phi_p));
      dp /= dp.norm_L2();

      Point2 p_proj(xn-d*dp.x, yn-d*dp.y);

      bool already_added = false;
      for(unsigned int m=0; m<added_points.size(); ++m)
      {
        if((p_proj-added_points[m]).norm_L2() < diag_min/5)
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
    else if(!is_interface)
    {
      Voronoi2D v;
      v.set_Center_Point(xn, yn);
      grid2voro[n].push_back(voro.size());
      voro.push_back(v);
    }
  }

  /* finally build the voronoi points from the list of projected points */
  for(unsigned int n=0; n<added_points.size(); ++n)
  {
    Point2 p_proj = added_points[n];
    Point2 dp = added_points_grad[n];

    /* add first point */
    Voronoi2D v;
    double xyz1 [] =
    {
      std::min(xmax, std::max(xmin, p_proj.x + band*dp.x)),
      std::min(ymax, std::max(ymin, p_proj.y + band*dp.y))
    };

    p4est_quadrant_t quad;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz1, quad, remote_matches);

    if(rank_found==p4est->mpirank)
    {
      p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
      p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
      p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
      double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];

      double qh = P4EST_QUADRANT_LEN(quad.level) / (double) P4EST_ROOT_LEN;
      double qx = quad.x / (double) P4EST_ROOT_LEN + tree_xmin;
      double qy = quad.y / (double) P4EST_ROOT_LEN + tree_ymin;
      p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

      p4est_locidx_t node = -1;
      if     (xyz1[0]<=qx+qh/2 && xyz1[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
      else if(xyz1[0]<=qx+qh/2 && xyz1[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
      else if(xyz1[0]> qx+qh/2 && xyz1[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
      else if(xyz1[0]> qx+qh/2 && xyz1[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];

      grid2voro[node].push_back(voro.size());
      v.set_Center_Point(xyz1[0], xyz1[1]);
      voro.push_back(v);
    }

    /* add second point */
    double xyz2 [] =
    {
      std::min(xmax, std::max(xmin, p_proj.x - band*dp.x)),
      std::min(ymax, std::max(ymin, p_proj.y - band*dp.y))
    };

    remote_matches.clear();
    rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz2, quad, remote_matches);

    if(rank_found==p4est->mpirank)
    {
      p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
      p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
      p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
      double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];

      double qh = P4EST_QUADRANT_LEN(quad.level) / (double) P4EST_ROOT_LEN;
      double qx = quad.x / (double) P4EST_ROOT_LEN + tree_xmin;
      double qy = quad.y / (double) P4EST_ROOT_LEN + tree_ymin;
      p4est_locidx_t quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;

      p4est_locidx_t node = -1;
      if     (xyz2[0]<=qx+qh/2 && xyz2[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
      else if(xyz2[0]<=qx+qh/2 && xyz2[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
      else if(xyz2[0]> qx+qh/2 && xyz2[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
      else if(xyz2[0]> qx+qh/2 && xyz2[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];

      grid2voro[node].push_back(voro.size());
      v.set_Center_Point(xyz2[0], xyz2[1]);
      voro.push_back(v);
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
          Point2 pm = voro[grid2voro[n][m]].get_Center_Point();
          v.x = pm.x; v.y = pm.y;
          buff_send_points[sharers[s]].push_back(v);
        }
      }
    }
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  /* send the data to remote processes */
  for(int r=0; r<p4est->mpisize; ++r)
  {
    if(send_to[r]==true)
    {
      MPI_Request req;
      MPI_Isend((void*)&buff_send_points[r][0], buff_send_points[r].size()*sizeof(voro_comm_t), MPI_BYTE, r, 2, p4est->mpicomm, &req);
      MPI_Request_free(&req);
    }
  }

  /* get local number of voronoi points for every processor */
  num_local_voro = voro.size();
  global_voro_offset[p4est->mpirank] = num_local_voro;
  MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &global_voro_offset[0], 1, MPI_INT, p4est->mpicomm);
  for(int r=1; r<p4est->mpisize; ++r)
  {
    global_voro_offset[r] += global_voro_offset[r-1];
  }

  global_voro_offset.insert(global_voro_offset.begin(), 0);

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

  int nb_recv = 0;
  for(int r=0; r<p4est->mpisize; ++r)
    if(recv_fr[r]) nb_recv++;

  /* now receive the data */
  while(nb_recv>0)
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
      };

      p4est_quadrant_t quad;
      std::vector<p4est_quadrant_t> remote_matches;
      int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);

      p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
      p4est_tree_t *tree = p4est_tree_array_index(p4est->trees, tree_idx);
      p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
      double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
      double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];

      double qh = P4EST_QUADRANT_LEN(quad.level) / (double) P4EST_ROOT_LEN;
      double qx = quad.x / (double) P4EST_ROOT_LEN + tree_xmin;
      double qy = quad.y / (double) P4EST_ROOT_LEN + tree_ymin;

      p4est_locidx_t quad_idx;
      if(rank_found==p4est->mpirank) quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;
      else                           quad_idx = quad.p.piggy3.local_num + p4est->local_num_quadrants;

      p4est_locidx_t node = -1;
      if     (xyz[0]<=qx+qh/2 && xyz[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mmm];
      else if(xyz[0]<=qx+qh/2 && xyz[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_mpm];
      else if(xyz[0]> qx+qh/2 && xyz[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_pmm];
      else if(xyz[0]> qx+qh/2 && xyz[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir::v_ppm];

      Voronoi2D v;
      v.set_Center_Point(xyz[0], xyz[1]);
      grid2voro[node].push_back(voro.size());
      voro.push_back(v);

      voro_ghost_local_num.push_back(buff_recv_points[n].local_num);
      voro_ghost_rank.push_back(sender_rank);
    }

    nb_recv--;
  }

  /* clear buffers */
  for(int r=0; r<p4est->mpisize; ++r)
  {
    buff_send_points[r].clear();
  }
  buff_send_points.clear();
  send_to.clear();
  recv_fr.clear();

  for(unsigned int n=0; n<num_local_voro; ++n)
  {
    /* find the cell to which this point belongs */
    Point2 pc;
    pc = voro[n].get_Center_Point();

    double xyz [] = {pc.x, pc.y};
    p4est_quadrant_t quad;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);

    /* check if the point is exactly a node */
    p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
    double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];

    double qh = P4EST_QUADRANT_LEN(quad.level) / (double) P4EST_ROOT_LEN;
    double qx = quad.x / (double) P4EST_ROOT_LEN + tree_xmin;
    double qy = quad.y / (double) P4EST_ROOT_LEN + tree_ymin;

    p4est_locidx_t quad_idx;
#ifdef CASL_THROWS
    if(rank_found==-1)
      throw std::invalid_argument("[CASL_ERROR]: PoissonSolverNodeBaseJump->compute_voronoi_mesh: found remote quadrant.");
#endif
    if(rank_found==p4est->mpirank) quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;
    else                           quad_idx = quad.p.piggy3.local_num + p4est->local_num_quadrants;

    std::vector<p4est_locidx_t> ngbd_quads;

    /* if exactly on a grid node */
    if( (fabs(xyz[0]-qx)<EPS || fabs(xyz[0]-(qx+qh))<EPS) &&
        (fabs(xyz[1]-qy)<EPS || fabs(xyz[1]-(qy+qh))<EPS) )
    {
      int dir = (fabs(xyz[0]-qx)<EPS ?
            (fabs(xyz[1]-qy)<EPS ? dir::v_mmm : dir::v_mpm)
          : (fabs(xyz[1]-qy)<EPS ? dir::v_pmm : dir::v_ppm) );
      p4est_locidx_t node = nodes->local_nodes[P4EST_CHILDREN*quad_idx + dir];

      p4est_locidx_t quad_idx;

      const my_p4est_cell_neighbors_t::quad_info_t* it;

      ngbd_n->find_neighbor_cell_of_node(node, -1, -1, quad_idx, tree_idx);
      if(quad_idx>=0)
      {
        ngbd_quads.push_back(quad_idx);
        for(it=ngbd_c->begin(quad_idx, dir::f_m00); it<ngbd_c->end(quad_idx, dir::f_m00); ++it)
          ngbd_quads.push_back(it->locidx);
        for(it=ngbd_c->begin(quad_idx, dir::f_0m0); it<ngbd_c->end(quad_idx, dir::f_0m0); ++it)
          ngbd_quads.push_back(it->locidx);
      }

      ngbd_n->find_neighbor_cell_of_node(node,  1, -1, quad_idx, tree_idx);
      if(quad_idx>=0)
      {
        ngbd_quads.push_back(quad_idx);
        for(it=ngbd_c->begin(quad_idx, dir::f_p00); it<ngbd_c->end(quad_idx, dir::f_p00); ++it)
          ngbd_quads.push_back(it->locidx);
        for(it=ngbd_c->begin(quad_idx, dir::f_0m0); it<ngbd_c->end(quad_idx, dir::f_0m0); ++it)
          ngbd_quads.push_back(it->locidx);
      }

      ngbd_n->find_neighbor_cell_of_node(node, -1,  1, quad_idx, tree_idx);
      if(quad_idx>=0)
      {
        ngbd_quads.push_back(quad_idx);
        for(it=ngbd_c->begin(quad_idx, dir::f_m00); it<ngbd_c->end(quad_idx, dir::f_m00); ++it)
          ngbd_quads.push_back(it->locidx);
        for(it=ngbd_c->begin(quad_idx, dir::f_0p0); it<ngbd_c->end(quad_idx, dir::f_0p0); ++it)
          ngbd_quads.push_back(it->locidx);
      }

      ngbd_n->find_neighbor_cell_of_node(node,  1,  1, quad_idx, tree_idx);
      if(quad_idx>=0)
      {
        ngbd_quads.push_back(quad_idx);
        for(it=ngbd_c->begin(quad_idx, dir::f_p00); it<ngbd_c->end(quad_idx, dir::f_p00); ++it)
          ngbd_quads.push_back(it->locidx);
        for(it=ngbd_c->begin(quad_idx, dir::f_0p0); it<ngbd_c->end(quad_idx, dir::f_0p0); ++it)
          ngbd_quads.push_back(it->locidx);
      }
    }
    /* the voronoi point is not a grid node */
    else
    {
      ngbd_quads.push_back(quad_idx);

      const my_p4est_cell_neighbors_t::quad_info_t* it;

      for(it=ngbd_c->begin(ngbd_quads[0], dir::f_m00); it<ngbd_c->end(ngbd_quads[0], dir::f_m00); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(ngbd_quads[0], dir::f_p00); it<ngbd_c->end(ngbd_quads[0], dir::f_p00); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(ngbd_quads[0], dir::f_0m0); it<ngbd_c->end(ngbd_quads[0], dir::f_0m0); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(ngbd_quads[0], dir::f_0p0); it<ngbd_c->end(ngbd_quads[0], dir::f_0p0); ++it)
        ngbd_quads.push_back(it->locidx);

      p4est_locidx_t n_idx;
      p4est_locidx_t q_idx;
      p4est_topidx_t t_idx;
      n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mmm];
      ngbd_n->find_neighbor_cell_of_node(n_idx, -1, -1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
      n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mpm];
      ngbd_n->find_neighbor_cell_of_node(n_idx, -1,  1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
      n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_pmm];
      ngbd_n->find_neighbor_cell_of_node(n_idx,  1, -1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);
      n_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_ppm];
      ngbd_n->find_neighbor_cell_of_node(n_idx,  1,  1, q_idx, t_idx); if(q_idx>=0) ngbd_quads.push_back(q_idx);

      int s = ngbd_quads.size();
      for(int i=1; i<s; ++i)
      {
        for(it=ngbd_c->begin(ngbd_quads[i], dir::f_m00); it<ngbd_c->end(ngbd_quads[i], dir::f_m00); ++it)
          ngbd_quads.push_back(it->locidx);
        for(it=ngbd_c->begin(ngbd_quads[i], dir::f_p00); it<ngbd_c->end(ngbd_quads[i], dir::f_p00); ++it)
          ngbd_quads.push_back(it->locidx);
        for(it=ngbd_c->begin(ngbd_quads[i], dir::f_0m0); it<ngbd_c->end(ngbd_quads[i], dir::f_0m0); ++it)
          ngbd_quads.push_back(it->locidx);
        for(it=ngbd_c->begin(ngbd_quads[i], dir::f_0p0); it<ngbd_c->end(ngbd_quads[i], dir::f_0p0); ++it)
          ngbd_quads.push_back(it->locidx);
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
            Point2 pm = voro[grid2voro[n_idx][m]].get_Center_Point();
            voro[n].push(grid2voro[n_idx][m], pm.x, pm.y);
          }
        }
      }
    }

    /* add the walls */
    if(is_quad_xmWall(p4est, quad.p.piggy3.which_tree, &quad)) voro[n].push(WALL_m00, pc.x-MAX(EPS, 2*(pc.x-xmin)), pc.y );
    if(is_quad_xpWall(p4est, quad.p.piggy3.which_tree, &quad)) voro[n].push(WALL_p00, pc.x+MAX(EPS, 2*(xmax-pc.x)), pc.y );
    if(is_quad_ymWall(p4est, quad.p.piggy3.which_tree, &quad)) voro[n].push(WALL_0m0, pc.x, pc.y-MAX(EPS, 2*(pc.y-ymin)));
    if(is_quad_ypWall(p4est, quad.p.piggy3.which_tree, &quad)) voro[n].push(WALL_0p0, pc.x, pc.y+MAX(EPS, 2*(ymax-pc.y)));

    /* finally, construct the partition */
    voro[n].construct_Partition();
  }

  ierr = VecCreateGhostVoronoiRhs(); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_compute_voronoi_mesh, 0, 0, 0, 0); CHKERRXX(ierr);

//  char name[1000];
//  sprintf(name, "/home/guittet/code/Output/p4est_jump/vtu/voronoi_%d.vtk", p4est->mpirank);
//  Voronoi2D::print_VTK_Format(voro, name);
//  Voronoi2D::print_VTK_Format(voro, err_voro, "err", name);
}



void PoissonSolverNodeBaseJump::setup_negative_laplace_matrix()
{
  preallocate_matrix();

  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

  for(unsigned int n=0; n<num_local_voro; ++n)
  {
    PetscInt global_n_idx = n+global_voro_offset[p4est->mpirank];

    Point2 pc = voro[n].get_Center_Point();
    if( (ABS(pc.x-xmin)<EPS || ABS(pc.x-xmax)<EPS ||
         ABS(pc.y-ymin)<EPS || ABS(pc.y-ymax)<EPS) &&
        bc->wallType(pc.x,pc.y)==DIRICHLET)
    {
      matrix_has_nullspace = false;
      ierr = MatSetValue(A, global_n_idx, global_n_idx, 1, ADD_VALUES); CHKERRXX(ierr);
      continue;
    }

    const std::vector<Voronoi2DPoint> *points;
    const std::vector<Point2> *partition;
    voro[n].get_Partition(points, partition);
    double phi_n = interp_phi(pc.x, pc.y);
    double mu_n;
    if(phi_n<0) mu_n = (*mu_m)(pc.x, pc.y);
    else        mu_n = (*mu_p)(pc.x, pc.y);

    double add_n = (*add)(pc.x, pc.y);
    if(add_n>EPS) matrix_has_nullspace = false;

    ierr = MatSetValue(A, global_n_idx, global_n_idx, voro[n].area()*add_n, ADD_VALUES); CHKERRXX(ierr);

    for(unsigned int l=0; l<points->size(); ++l)
    {
      if((*points)[l].n>=0)
      {
        /* regular point */
        Point2 pl = (*points)[l].p;
        int k = (l+partition->size()-1) % partition->size();
        double s = ((*partition)[k]-(*partition)[l]).norm_L2();
        double d = (pc - pl).norm_L2();
        double phi_l = interp_phi(pl.x, pl.y);
        double mu_l;
        if(phi_l<0) mu_l = (*mu_m)(pl.x, pl.y);
        else        mu_l = (*mu_p)(pl.x, pl.y);

        double mu_harmonic = 2*mu_n*mu_l/(mu_n + mu_l);

        PetscInt global_l_idx;
        if((unsigned int)(*points)[l].n<num_local_voro) global_l_idx = (*points)[l].n + global_voro_offset[p4est->mpirank];
        else                                            global_l_idx = voro_ghost_local_num[(*points)[l].n-num_local_voro] + global_voro_offset[voro_ghost_rank[(*points)[l].n-num_local_voro]];

        ierr = MatSetValue(A, global_n_idx, global_n_idx,  s*mu_harmonic/d, ADD_VALUES); CHKERRXX(ierr);
        ierr = MatSetValue(A, global_n_idx, global_l_idx, -s*mu_harmonic/d, ADD_VALUES); CHKERRXX(ierr);
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
    if (A_null_space == NULL) // pun not intended!
    {
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, PETSC_NULL, &A_null_space); CHKERRXX(ierr);
    }
    ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);
}

void PoissonSolverNodeBaseJump::setup_negative_laplace_rhsvec()
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);

  double *rhs_p;
  ierr = VecGetArray(rhs, &rhs_p); CHKERRXX(ierr);

  for(unsigned int n=0; n<num_local_voro; ++n)
  {
    Point2 pc = voro[n].get_Center_Point();
    if( (ABS(pc.x-xmin)<EPS || ABS(pc.x-xmax)<EPS ||
         ABS(pc.y-ymin)<EPS || ABS(pc.y-ymax)<EPS ) &&
        bc->wallType(pc.x,pc.y)==DIRICHLET)
    {
      rhs_p[n] = bc->wallValue(pc.x, pc.y);
      continue;
    }

    const std::vector<Voronoi2DPoint> *points;
    const std::vector<Point2> *partition;
    voro[n].get_Partition(points, partition);

    double p_n = interp_phi(pc.x, pc.y);
    double mu_n;

    if(p_n<0)
    {
      rhs_p[n] = this->rhs_m(pc.x, pc.y);
      mu_n     = (*mu_m)(pc.x, pc.y);
    }
    else
    {
      rhs_p[n] = this->rhs_p(pc.x, pc.y);
      mu_n     = (*mu_p)(pc.x, pc.y);
    }

    rhs_p[n] *= voro[n].area();

    for(unsigned int l=0; l<points->size(); ++l)
    {
      int k = (l+partition->size()-1) % partition->size();
      double s = ((*partition)[k]-(*partition)[l]).norm_L2();
      if((*points)[l].n>=0)
      {
        Point2 pl = (*points)[l].p;
        double p_l = interp_phi(pl.x, pl.y);
        if(p_n*p_l<0)
        {
          Point2 norm = p_n<0 ? pl - pc : pc - pl;
          double d = norm.norm_L2();
          norm /= d;
          double mu_l;
          if(p_l<0) mu_l = (*mu_m)(pl.x, pl.y);
          else      mu_l = (*mu_p)(pl.x, pl.y);
          Point2 p_ln = (pc+pl)/2;

          double mu_harmonic = 2*mu_n*mu_l/(mu_n + mu_l);

          rhs_p[n] += s*mu_harmonic/d * SIGN(p_n) * (*u_jump)(p_ln.x, p_ln.y);
          rhs_p[n] -= mu_harmonic/mu_l * s/2 * (*mu_grad_u_jump)(p_ln.x, p_ln.y);
        }
      }
      else /* wall with neumann */
      {
        double x_tmp = pc.x;
        double y_tmp = pc.y;

        /* perturb the corners to differentiate between the edges of the domain ... otherwise 1st order only at the corners */
        if(pc.y==ymin && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp += 2*EPS;
        if(pc.y==ymax && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp -= 2*EPS;
        if(pc.x==xmin && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp += 2*EPS;
        if(pc.x==xmax && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp -= 2*EPS;

        rhs_p[n] += s*mu_n * bc->wallValue(x_tmp, y_tmp);
      }
    }
  }

  ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

  if (matrix_has_nullspace)
    ierr = MatNullSpaceRemove(A_null_space, rhs, NULL); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);
}




void PoissonSolverNodeBaseJump::interpolate_solution_from_voronoi_to_tree(Vec solution)
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_interpolate_to_tree, phi, sol_voro, solution, 0); CHKERRXX(ierr);

  double *phi_p, *sol_voro_p, *solution_p;
  ierr = VecGetArray(phi     , &phi_p     ); CHKERRXX(ierr);
  ierr = VecGetArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);
  ierr = VecGetArray(solution, &solution_p); CHKERRXX(ierr);

//  /* for debugging, compute the error on the voronoi mesh */
//  double err = 0;
//  for(unsigned int n=0; n<num_local_voro; ++n)
//  {
//    Point2 pc = voro[n].get_Center_Point();

////    double phi_n = interp_phi(pc.x, pc.y);
//    double u_ex;
//    u_ex = cos(pc.x)*sin(pc.y);
////    if(phi_n<0) u_ex = exp(pc.x);
////    else        u_ex = cos(pc.x)*sin(pc.y);

//    err = std::max(err, ABS(u_ex - sol_voro_p[n]));
//  }
//  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
//  PetscPrintf(p4est->mpicomm, "Error : %g\n", err);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
#ifdef P4_TO_P8
    Point3 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes), node_z_fr_n(n, p4est, nodes));
#else
    Point2 pn(node_x_fr_n(n, p4est, nodes), node_y_fr_n(n, p4est, nodes));
#endif

    /* first check if the node is a voronoi point */
    if(grid2voro[n].size()>0)
    {
      bool voro_point = false;
      for(unsigned int m=0; m<grid2voro[n].size(); ++m)
      {
        Point2 pm = voro[grid2voro[n][m]].get_Center_Point();
        if((pn-pm).norm_L2()<EPS)
        {
          solution_p[n] = sol_voro_p[grid2voro[n][m]];
          voro_point = true;
          break;
        }
      }
      if(voro_point) continue;
    }

    /* if not a grid point, gather all the neighbor voro points and find the
     * three closest with the same sign for phi */
    p4est_locidx_t quad_idx;
    p4est_topidx_t tree_idx;

    const my_p4est_cell_neighbors_t::quad_info_t* it;

    std::vector<p4est_locidx_t> ngbd_quads;

    ngbd_n->find_neighbor_cell_of_node(n, -1, -1, quad_idx, tree_idx);
    if(quad_idx>=0)
    {
      ngbd_quads.push_back(quad_idx);
      for(it=ngbd_c->begin(quad_idx, dir::f_m00); it<ngbd_c->end(quad_idx, dir::f_m00); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(quad_idx, dir::f_0m0); it<ngbd_c->end(quad_idx, dir::f_0m0); ++it)
        ngbd_quads.push_back(it->locidx);
    }

    ngbd_n->find_neighbor_cell_of_node(n,  1, -1, quad_idx, tree_idx);
    if(quad_idx>=0)
    {
      ngbd_quads.push_back(quad_idx);
      for(it=ngbd_c->begin(quad_idx, dir::f_p00); it<ngbd_c->end(quad_idx, dir::f_p00); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(quad_idx, dir::f_0m0); it<ngbd_c->end(quad_idx, dir::f_0m0); ++it)
        ngbd_quads.push_back(it->locidx);
    }

    ngbd_n->find_neighbor_cell_of_node(n, -1,  1, quad_idx, tree_idx);
    if(quad_idx>=0)
    {
      ngbd_quads.push_back(quad_idx);
      for(it=ngbd_c->begin(quad_idx, dir::f_m00); it<ngbd_c->end(quad_idx, dir::f_m00); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(quad_idx, dir::f_0p0); it<ngbd_c->end(quad_idx, dir::f_0p0); ++it)
        ngbd_quads.push_back(it->locidx);
    }

    ngbd_n->find_neighbor_cell_of_node(n,  1,  1, quad_idx, tree_idx);
    if(quad_idx>=0)
    {
      ngbd_quads.push_back(quad_idx);
      for(it=ngbd_c->begin(quad_idx, dir::f_p00); it<ngbd_c->end(quad_idx, dir::f_p00); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(quad_idx, dir::f_0p0); it<ngbd_c->end(quad_idx, dir::f_0p0); ++it)
        ngbd_quads.push_back(it->locidx);
    }

    /* now find the two voronoi points closest to the node */
    double phi_n = phi_p[n];
    double di[] = {DBL_MAX, DBL_MAX, DBL_MAX};
    unsigned int ni[] = {UINT_MAX, UINT_MAX, UINT_MAX};

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
            Point2 pm = voro[grid2voro[n_idx][m]].get_Center_Point();
            double xyz[] = {pm.x, pm.y};
            p4est_quadrant_t quad;
            std::vector<p4est_quadrant_t> remote_matches;

            int rank = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);
            if(rank!=-1)
            {
              double phi_m = interp_phi(pm.x, pm.y);
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
            }
          }
        }
      }
    }

    Point2 p0(voro[ni[0]].get_Center_Point());
    Point2 p1(voro[ni[1]].get_Center_Point());

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
            Point2 pm = voro[grid2voro[n_idx][m]].get_Center_Point();
            double xyz[] = {pm.x, pm.y};
            p4est_quadrant_t quad;
            std::vector<p4est_quadrant_t> remote_matches;

            int rank = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);
            if(rank!=-1)
            {
              double phi_m = interp_phi(pm.x, pm.y);
              if(phi_m*phi_n>=0)
              {
                double d = (pm-pn).norm_L2();

                if(d<di[2] && ABS((p0-p1).normalize().cross((pm-p1).normalize())) > sin(PI/5))
                {
                  ni[2]=grid2voro[n_idx][m]; di[2]=d;
                }
              }
            }
          }
        }
      }
    }

    Point2 p2(voro[ni[2]].get_Center_Point());

    /* make sure we found 3 points */
    if(di[0]==DBL_MAX || di[1]==DBL_MAX || di[2]==DBL_MAX)
    {
      std::cerr << "PoissonSolverNodeBaseJump->interpolate_solution_from_voronoi_to_tree: not enough points found." << std::endl;
      solution_p[n] = sol_voro_p[ni[0]];
      continue;
    }

    if(ni[0]==ni[1] || ni[0]==ni[2] || ni[1]==ni[2])
      std::cerr << "PoissonSolverNodeBaseJump->interpolate_solution_from_voronoi_to_tree: point is double !" << std::endl;

    double f0 = sol_voro_p[ni[0]];
    double f1 = sol_voro_p[ni[1]];
    double f2 = sol_voro_p[ni[2]];

    double det = p0.x*p1.y + p1.x*p2.y + p2.x*p0.y - p1.x*p0.y - p2.x*p1.y - p0.x*p2.y;

#ifdef CASL_THROWS
    if(ABS(det)<EPS)
      throw std::invalid_argument("[CASL_ERROR]: interpolation_Voronoi: could not invert system ...");
#endif

    double c0 = ( (p1.y* 1- 1*p2.y)*f0 + ( 1*p2.y-p0.y* 1)*f1 + (p0.y* 1- 1*p1.y)*f2 ) / det;
    double c1 = ( ( 1*p2.x-p1.x* 1)*f0 + (p0.x* 1- 1*p2.x)*f1 + ( 1*p1.x-p0.x* 1)*f2 ) / det;
    double c2 = ( (p1.x*p2.y-p2.x*p1.y)*f0 + (p2.x*p0.y-p0.x*p2.y)*f1 + (p0.x*p1.y-p1.x*p0.y)*f2 ) / det;

    solution_p[n] = c0*pn.x + c1*pn.y + c2;
  }

  ierr = VecRestoreArray(phi     , &phi_p     ); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(solution, &solution_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_interpolate_to_tree, phi, sol_voro, solution, 0); CHKERRXX(ierr);
}
