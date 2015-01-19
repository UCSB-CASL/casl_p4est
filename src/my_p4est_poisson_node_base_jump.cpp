#include <src/my_p4est_poisson_node_base_jump.h>
#include <src/my_p4est_refine_coarsen.h>

#include <src/petsc_compatibility.h>
#include <src/CASL_math.h>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_PoissonSolverNodeBaseJumpd_matrix_preallocation;
extern PetscLogEvent log_PoissonSolverNodeBaseJumpd_matrix_setup;
extern PetscLogEvent log_PoissonSolverNodeBaseJumpd_rhsvec_setup;
extern PetscLogEvent log_PoissonSolverNodeBaseJumpd_KSPSolve;
extern PetscLogEvent log_PoissonSolverNodeBaseJumpd_solve;
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
    A(NULL),
    matrix_has_nullspace(NULL), is_matrix_computed(NULL)
{
  // compute global numbering of nodes
  global_node_offset.resize(p4est->mpisize+1, 0);

  // Calculate the global number of points
  for (int r = 0; r<p4est->mpisize; ++r)
    global_node_offset[r+1] = global_node_offset[r] + (PetscInt)nodes->global_owned_indeps[r];

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
  if (A             != NULL) ierr = MatDestroy(A);                      CHKERRXX(ierr);
  if (ksp           != NULL) ierr = KSPDestroy(ksp);                    CHKERRXX(ierr);
}


void PoissonSolverNodeBaseJump::set_phi(Vec phi)
{
  this->phi = phi;
}


void PoissonSolverNodeBaseJump::preallocate_matrix()
{  
  /* enable logging for the preallocation */
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBaseJumpd_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = global_node_offset[p4est->mpisize];
  PetscInt num_owned_local  = (PetscInt)(nodes->num_owned_indeps);

  if (A != NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  /* set up the matrix */
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

//  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
//  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBaseJumpd_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void PoissonSolverNodeBaseJump::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBaseJumpd_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);

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

  // setup rhs
  setup_negative_laplace_rhsvec();

  // Solve the system
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBaseJumpd_KSPSolve, ksp, rhs_, solution, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_PoissonSolverNodeBaseJumpd_KSPSolve, ksp, rhs_, solution, 0); CHKERRXX(ierr);

  // update ghosts
  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBaseJumpd_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);
}


void PoissonSolverNodeBaseJump::compute_voronoi_mesh()
{
  PetscErrorCode ierr;
  grid2voro.resize(nodes->indep_nodes.elem_count);

  voro.resize(0);
  std::vector<Point2> added_points;

  double band = diag_min/5;
  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; ++n)
  {
    double p_00, p_m0, p_p0, p_0m, p_0p;
    (*ngbd_n)[n].ngbd_with_quadratic_interpolation(phi_p, p_00, p_m0, p_p0, p_0m, p_0p);

    p4est_indep_t *node = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    p4est_topidx_t tree_id = node->p.piggy3.which_tree;

    p4est_topidx_t v_mm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_id + 0];

    double tree_xmin = p4est->connectivity->vertices[3*v_mm + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_mm + 1];
#ifdef P4_TO_P8
    double tree_zmin = p4est->connectivity->vertices[3*v_mm + 2];
#endif

    double xn = node_x_fr_n(node) + tree_xmin;
    double yn = node_y_fr_n(node) + tree_ymin;

    if(p_00*p_m0<=0 || p_00*p_p0<=0 || p_00*p_0m<=0 || p_00*p_0p<=0)
    {
      double d = phi_p[n];
      Point2 dp((*ngbd_n)[n].dx_central(phi_p), (*ngbd_n)[n].dy_central(phi_p));
      dp /= dp.norm_L2();

      Point2 p_proj(xn-d*dp.x, yn-d*dp.y);

      bool already_added = false;
      for(unsigned int m=0; m<added_points.size(); ++m)
      {
        double x_tmp = p_proj.x;
        double y_tmp = p_proj.y;

        if(sqrt(SQR(added_points[m].x-x_tmp) + SQR(added_points[m].y-y_tmp)) < diag_min/5)
        {
          already_added = true;
          break;
        }
      }

      if(!already_added)
      {
        added_points.push_back(p_proj);

        /* add first point */
        Voronoi2D v;
        double xyz1 [] =
        {
          std::min(xmax, std::max(xmin, p_proj.x + band*dp.x)),
          std::min(ymax, std::max(ymin, p_proj.y + band*dp.y))
        };
        v.set_Center_Point(xyz1[0], xyz1[1]);

        p4est_quadrant_t quad;
        std::vector<p4est_quadrant_t> remote_matches;
        int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz1, quad, remote_matches);
        if(rank_found!=p4est->mpirank)
          throw std::invalid_argument("compute_voronoi_mesh: found quadrant in remote process.");

        p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
        p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
        p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
        double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
        double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];

        double qh = P4EST_QUADRANT_LEN(quad.level) / (double) P4EST_ROOT_LEN;
        double qx = quad.x / (double) P4EST_ROOT_LEN + tree_xmin;
        double qy = quad.y / (double) P4EST_ROOT_LEN + tree_ymin;

        p4est_locidx_t node = -1;
        if     (xyz1[0]<=qx+qh/2 && xyz1[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*(quad.p.piggy3.local_num+tree->quadrants_offset) + dir::v_mmm];
        else if(xyz1[0]<=qx+qh/2 && xyz1[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*(quad.p.piggy3.local_num+tree->quadrants_offset) + dir::v_mpm];
        else if(xyz1[0]> qx+qh/2 && xyz1[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*(quad.p.piggy3.local_num+tree->quadrants_offset) + dir::v_pmm];
        else if(xyz1[0]> qx+qh/2 && xyz1[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*(quad.p.piggy3.local_num+tree->quadrants_offset) + dir::v_ppm];

        if(node<nodes->num_owned_indeps)
        {
          grid2voro[node].push_back(voro.size());
          voro.push_back(v);
        }

        /* add second point */
        double xyz2 [] =
        {
          std::min(xmax, std::max(xmin, p_proj.x - band*dp.x)),
          std::min(ymax, std::max(ymin, p_proj.y - band*dp.y))
        };
        v.set_Center_Point(xyz2[0], xyz2[1]);

        remote_matches.clear();
        rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz2, quad, remote_matches);

        tree_idx = quad.p.piggy3.which_tree;
        tree = p4est_tree_array_index(p4est->trees, tree_idx);
        v_mmm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
        tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
        tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];

        qh = P4EST_QUADRANT_LEN(quad.level) / (double) P4EST_ROOT_LEN;
        qx = quad.x / (double) P4EST_ROOT_LEN + tree_xmin;
        qy = quad.y / (double) P4EST_ROOT_LEN + tree_ymin;

        node = -1;
        if     (xyz2[0]<=qx+qh/2 && xyz2[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*(quad.p.piggy3.local_num+tree->quadrants_offset) + dir::v_mmm];
        else if(xyz2[0]<=qx+qh/2 && xyz2[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*(quad.p.piggy3.local_num+tree->quadrants_offset) + dir::v_mpm];
        else if(xyz2[0]> qx+qh/2 && xyz2[1]<=qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*(quad.p.piggy3.local_num+tree->quadrants_offset) + dir::v_pmm];
        else if(xyz2[0]> qx+qh/2 && xyz2[1]> qy+qh/2) node = nodes->local_nodes[P4EST_CHILDREN*(quad.p.piggy3.local_num+tree->quadrants_offset) + dir::v_ppm];

        if(node<nodes->num_owned_indeps)
        {
          grid2voro[node].push_back(voro.size());
          voro.push_back(v);
        }
      }
    }
    else
    {
      Voronoi2D v;
      v.set_Center_Point(xn, yn);
      grid2voro[n].push_back(voro.size());
      voro.push_back(v);
    }
  }



  for(unsigned int n=0; n<voro.size(); ++n)
  {
    /* find the cell to which this point belongs */
    Point2 pc;
    pc = voro[n].get_Center_Point();

    double xyz [] = {pc.x, pc.y};
    p4est_quadrant_t quad;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz, quad, remote_matches);
    if(rank!=p4est->mpirank)
      throw std::invalid_argument("compute_voronoi_mesh: found remote quadrant.");

    /* check if the point is exactly a node */
    p4est_topidx_t tree_idx = quad.p.piggy3.which_tree;
    p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, tree_idx);
    p4est_topidx_t v_mmm = p4est->connectivity->tree_to_vertex[P4EST_CHILDREN*tree_idx + 0];
    double tree_xmin = p4est->connectivity->vertices[3*v_mmm + 0];
    double tree_ymin = p4est->connectivity->vertices[3*v_mmm + 1];

    double qh = P4EST_QUADRANT_LEN(quad.level) / (double) P4EST_ROOT_LEN;
    double qx = quad.x / (double) P4EST_ROOT_LEN + tree_xmin;
    double qy = quad.y / (double) P4EST_ROOT_LEN + tree_ymin;

    std::vector<p4est_locidx_t> ngbd_quads;

    /* if exactly on a grid node */

    if( (fabs(xyz[0]-qx)<EPS || fabs(xyz[0]-(qx+qh))<EPS) &&
        (fabs(xyz[1]-qy)<EPS || fabs(xyz[1]-(qy+qh))<EPS) )
    {
      std::cout << "type 1" << std::endl;
      int dir = (fabs(xyz[0]-qx)<EPS ?
            (fabs(xyz[1]-qy)<EPS ? dir::v_mmm : dir::v_mpm)
          : (fabs(xyz[1]-qy)<EPS ? dir::v_pmm : dir::v_ppm) );
      p4est_locidx_t node = nodes->local_nodes[P4EST_CHILDREN*(quad.p.piggy3.local_num+tree->quadrants_offset) + dir];

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
      std::cout << "type 2" << std::endl;
      ngbd_quads.push_back(quad.p.piggy3.local_num + tree->quadrants_offset);

      const my_p4est_cell_neighbors_t::quad_info_t* it;

      for(it=ngbd_c->begin(ngbd_quads[0], dir::f_m00); it<ngbd_c->end(ngbd_quads[0], dir::f_m00); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(ngbd_quads[0], dir::f_p00); it<ngbd_c->end(ngbd_quads[0], dir::f_p00); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(ngbd_quads[0], dir::f_0m0); it<ngbd_c->end(ngbd_quads[0], dir::f_0m0); ++it)
        ngbd_quads.push_back(it->locidx);
      for(it=ngbd_c->begin(ngbd_quads[0], dir::f_0p0); it<ngbd_c->end(ngbd_quads[0], dir::f_0p0); ++it)
        ngbd_quads.push_back(it->locidx);

//      for(unsigned int i=0; i<ngbd_quads.size(); ++i)
//        std::cout << ngbd_quads[i] << std::endl;

      p4est_locidx_t node_idx;
      p4est_locidx_t quad_idx;
      node_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mmm];
      ngbd_n->find_neighbor_cell_of_node(node_idx, -1, -1, quad_idx, tree_idx); if(quad_idx>=0) ngbd_quads.push_back(quad_idx);
      node_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_mpm];
      ngbd_n->find_neighbor_cell_of_node(node_idx, -1,  1, quad_idx, tree_idx); if(quad_idx>=0) ngbd_quads.push_back(quad_idx);
      node_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_pmm];
      ngbd_n->find_neighbor_cell_of_node(node_idx,  1, -1, quad_idx, tree_idx); if(quad_idx>=0) ngbd_quads.push_back(quad_idx);
      node_idx = nodes->local_nodes[ngbd_quads[0]*P4EST_CHILDREN + dir::v_ppm];
      ngbd_n->find_neighbor_cell_of_node(node_idx,  1,  1, quad_idx, tree_idx); if(quad_idx>=0) ngbd_quads.push_back(quad_idx);
    }

    /* now create the list of nodes */
    for(unsigned int k=0; k<ngbd_quads.size(); ++k)
    {
//      std::cout << k << " : " << ngbd_quads[k] << std::endl;
      for(int dir=0; dir<P4EST_CHILDREN; ++dir)
      {
        p4est_locidx_t node_idx = nodes->local_nodes[P4EST_CHILDREN*ngbd_quads[k] + dir];
        for(unsigned int m=0; m<grid2voro[node_idx].size(); ++m)
        {
          if(grid2voro[node_idx][m] != n)
          {
            Point2 pm = voro[grid2voro[node_idx][m]].get_Center_Point();
            voro[n].push(grid2voro[node_idx][m], pm.x, pm.y);
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

  Voronoi2D::print_VTK_Format(voro, "/home/guittet/code/Output/p4est_jump/vtu/voronoi.vtk");


  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);
}



void PoissonSolverNodeBaseJump::setup_negative_laplace_matrix()
{

}

void PoissonSolverNodeBaseJump::setup_negative_laplace_rhsvec()
{

}
