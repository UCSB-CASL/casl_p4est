#ifdef P4_TO_P8
#include "my_p8est_poisson_cells.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
#else
#include "my_p4est_poisson_cells.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/cube2.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <algorithm>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_poisson_cells_matrix_preallocation;
extern PetscLogEvent log_my_p4est_poisson_cells_matrix_setup;
extern PetscLogEvent log_my_p4est_poisson_cells_update_matrix_diag_only;
extern PetscLogEvent log_my_p4est_poisson_cells_rhsvec_setup;
extern PetscLogEvent log_my_p4est_poisson_cells_solve;
extern PetscLogEvent log_my_p4est_poisson_cells_KSPSolve;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bcstrength 1.0

my_p4est_poisson_cells_t::my_p4est_poisson_cells_t(const my_p4est_cell_neighbors_t *ngbd_c,
                                                   const my_p4est_node_neighbors_t *ngbd_n)
  : ngbd_c(ngbd_c), ngbd_n(ngbd_n),
    p4est(ngbd_c->p4est), nodes(ngbd_n->nodes), ghost(ngbd_c->ghost), myb(ngbd_c->myb),
    mu(1.),
    is_matrix_ready(false), only_diag_is_modified(false),
    desired_diag_locally_built(false),
    ksp_is_set_from_options(false), pc_is_set_from_options(false),
    matrix_has_nullspace(false),
    bc(NULL),
    nullspace_use_fixed_point(false),
    A(NULL), A_null_space(NULL),
    null_space(NULL),
    current_diag(NULL), desired_diag(NULL), rhs(NULL), phi(NULL)
{
  // set up the KSP solver
  ierr = KSPCreate(p4est->mpicomm, &ksp); CHKERRXX(ierr);

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  // compute grid parameters
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  for(int dir=0; dir<P4EST_DIM; ++dir)
  {
    xyz_min[dir] = p4est->connectivity->vertices[3*vm + dir];
    xyz_max[dir] = p4est->connectivity->vertices[3*vp + dir];
    dxyz_min[dir] = (xyz_max[dir]-xyz_min[dir]) / pow(2.,(double) data->max_lvl);
  }

#ifdef P4_TO_P8
  d_min = MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
  diag_min = sqrt(SQR(dxyz_min[0]) + SQR(dxyz_min[1]) + SQR(dxyz_min[2]));
#else
  d_min = MIN(dxyz_min[0], dxyz_min[1]);
  diag_min = sqrt(SQR(dxyz_min[0]) + SQR(dxyz_min[1]));
#endif

  ierr = reset_current_diag(); CHKERRXX(ierr);
}

my_p4est_poisson_cells_t::~my_p4est_poisson_cells_t()
{
  if (A             != NULL)  { ierr = MatDestroy(A);                      CHKERRXX(ierr); }
  if (A_null_space  != NULL)  { ierr = MatNullSpaceDestroy (A_null_space); CHKERRXX(ierr); }
  if (current_diag  != NULL)  { ierr = VecDestroy(current_diag);           CHKERRXX(ierr); }
  if (desired_diag_locally_built && (desired_diag  != NULL))
                              { ierr = VecDestroy(desired_diag);           CHKERRXX(ierr); }
  if (null_space    != NULL)  { ierr = VecDestroy(null_space);             CHKERRXX(ierr); }
  if (ksp           != NULL)  { ierr = KSPDestroy(ksp);                    CHKERRXX(ierr); }
}

void my_p4est_poisson_cells_t::preallocate_matrix()
{
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_poisson_cells_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = p4est->global_num_quadrants;
  PetscInt num_owned_local  = p4est->local_num_quadrants;

  std::vector<p4est_quadrant_t> ngbd;

  if (A != NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  // set up the matrix
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  std::vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);
  double *phi_p;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  const p4est_locidx_t *q2n = nodes->local_nodes;
  std::vector<p4est_locidx_t> indices;

  double xyz_q[P4EST_DIM];

  for (p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);
    for (size_t q=0; q<tree->quadrants.elem_count; q++)
    {
      const p4est_quadrant_t *quad  = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      /* check if we are in the Omega^- domain */
      double phi_c = 0;
      bool all_pos = true;
      for (short i=0; i<P4EST_CHILDREN; i++)
      {
        double tmp = phi_p[q2n[quad_idx*P4EST_CHILDREN + i]];
        phi_c += tmp;
        all_pos = all_pos && (tmp>0);
      }
      phi_c /= (double)P4EST_CHILDREN;

      quad_xyz_fr_q(quad_idx, tree_idx, p4est, ghost, xyz_q);
      if((bc->interfaceType(xyz_q)==DIRICHLET && phi_c > 0) || (bc->interfaceType(xyz_q)==NEUMANN && all_pos))
        continue;

      indices.resize(0);

      /*
     * Check for neighboring cells:
     * 1) If they exist and are local quads, increment d_nnz[n]
     * 2) If they exist but are not local quads, increment o_nnz[n]
     * 3) If they do not exist, simply skip
     */
      for(int dir=0; dir<P4EST_FACES; ++dir)
      {
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);

        if(ngbd.size()==1 && ngbd[0].level<quad->level)
        {
          p4est_locidx_t q_tmp = ngbd[0].p.piggy3.local_num;
          p4est_topidx_t t_tmp = ngbd[0].p.piggy3.which_tree;

          /* no need to add this one to "indices" since it can't be found with a search in another direction */
          if(q_tmp<num_owned_local) d_nnz[quad_idx]++;
          else                      o_nnz[quad_idx]++;

          ngbd.resize(0);
          ngbd_c->find_neighbor_cells_of_cell(ngbd, q_tmp, t_tmp, dir%2==0 ? dir+1 : dir-1);
          for(unsigned int m=0; m<ngbd.size(); ++m)
          {
            if(ngbd[m].p.piggy3.local_num!=quad_idx && std::find(indices.begin(), indices.end(), ngbd[m].p.piggy3.local_num)==indices.end())
            {
              indices.push_back(ngbd[m].p.piggy3.local_num);
              if(ngbd[m].p.piggy3.local_num<num_owned_local) d_nnz[quad_idx]++;
              else                                           o_nnz[quad_idx]++;
            }
          }
        }
        else
        {
          for(unsigned int m=0; m<ngbd.size(); ++m)
          {
            if(std::find(indices.begin(), indices.end(), ngbd[m].p.piggy3.local_num)==indices.end())
            {
              indices.push_back(ngbd[m].p.piggy3.local_num);
              if(ngbd[m].p.piggy3.local_num<num_owned_local) d_nnz[quad_idx]++;
              else                                           o_nnz[quad_idx]++;
            }
          }
        }
      }
    }
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_cells_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_cells_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_cells_solve, A, rhs, ksp, 0); CHKERRXX(ierr);

#ifdef CASL_THROWS
  if(bc == NULL) throw std::domain_error("[CASL_ERROR]: the boundary conditions have not been set.");

  {
    PetscInt sol_size;
    ierr = VecGetLocalSize(solution, &sol_size); CHKERRXX(ierr);
    if (sol_size != p4est->local_num_quadrants){
      std::ostringstream oss;
      oss << "[CASL_ERROR]: solution vector must be preallocated and locally have the same size as num_owned_indeps"
          << "solution.local_size = " << sol_size << " p4est->local_num_quadrants = " << p4est->local_num_quadrants << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
#endif

  // set a local phi if not was given
  bool local_phi = false;
  if(phi == NULL)
  {
    local_phi = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi); CHKERRXX(ierr);
    ierr = VecSet(phi, -1.); CHKERRXX(ierr);
  }

  /*
   * Here we set the matrix, ksp, and pc.
   * If required, the matrix is assembled and the Krylov solver is initialized based on that matrix.
   * If the matrix is already assembled, we reset the Krylo
   * If the matrix has not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  if (!is_matrix_ready)
  {
    if (only_diag_is_modified)
      update_matrix_diag_only();
    else
    {
      reset_current_diag();
      setup_negative_laplace_matrix();
    }
  }
  ierr = KSPSetOperators(ksp, A, A, SAME_NONZERO_PATTERN); CHKERRXX(ierr);
  /* [Raphael Egan:] Starting from version 3.5, the last argument in KSPSetOperators became
   * irrelevant and is now simply disregarded in the above call. The matrices now keep track
   * of changes to their values and/or to their nonzero pattern by themselves. If no
   * modification was made to the matrix, the ksp environment can figure it out and knows
   * that the current preconditioner is still valid, thus it won't be recomputed.
   * If one desires to force reusing the current preconditioner EVEN IF a modification was
   * made to the matrix, one needs to call
   * ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE); CHKERRXX(ierr);
   * before the subsequent call to KSPSolve().
   * I have decided not to enforce that...
   */

  // setup rhs
  setup_negative_laplace_rhsvec();

  P4EST_ASSERT(ksp != NULL);
  // set ksp type
  KSPType ksp_type_as_such;
  ierr = KSPGetType(ksp, &ksp_type_as_such); CHKERRXX(ierr);
  if(ksp_type_as_such != ksp_type)
  {
    ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);
  }
  PetscReal ksp_tolerance;
  ierr = KSPGetTolerances(ksp, &ksp_tolerance, NULL, NULL, NULL); CHKERRXX(ierr);
  if(ksp_tolerance > 1.05e-12) // 1.05e-12 instead of 1e-12 to avoid floating-point arithmetics errors
  {
    ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  }
  PetscBool ksp_initial_guess;
  ierr = KSPGetInitialGuessNonzero(ksp, &ksp_initial_guess); CHKERRXX(ierr);
  if (use_nonzero_initial_guess != ksp_initial_guess)
  {
    ierr = KSPSetInitialGuessNonzero(ksp, ((use_nonzero_initial_guess)? PETSC_TRUE: PETSC_FALSE)); CHKERRXX(ierr);
  }
  if(!ksp_is_set_from_options)
  {
    ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);
    ksp_is_set_from_options = true;
  }

  // set pc type
  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRXX(ierr);
  P4EST_ASSERT(pc != NULL);
  PCType pc_type_as_such;
  ierr = PCGetType(pc, &pc_type_as_such); CHKERRXX(ierr);
  if(pc_type_as_such != pc_type)
  {
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
      if (matrix_has_nullspace && !nullspace_use_fixed_point){
        ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
      }
    }
  }

  if(!pc_is_set_from_options)
  {
    ierr = PCSetFromOptions(pc); CHKERRXX(ierr);
    pc_is_set_from_options = true;
  }

  // Solve the system
  ierr = PetscLogEventBegin(log_my_p4est_poisson_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd(log_my_p4est_poisson_cells_KSPSolve, solution, rhs, ksp, 0); CHKERRXX(ierr);

  // get rid of local stuff
  if(local_phi)
  {
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    phi = NULL;
  }

  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_cells_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_cells_t::setup_negative_laplace_matrix()
{
  preallocate_matrix();

  matrix_has_nullspace = true;
  double *null_space_p;
  if(!nullspace_use_fixed_point)
  {
    if(null_space != NULL)
    {
      ierr = VecDestroy(null_space); CHKERRXX(ierr);
    }
    ierr = VecDuplicate(rhs, &null_space); CHKERRXX(ierr);
    ierr = VecGetArray(null_space, &null_space_p); CHKERRXX(ierr);
  }

  fixed_value_idx_g = p4est->global_num_quadrants;

  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_cells_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);
  double *phi_p, *current_diag_p;
  const double* desired_diag_p;

  ierr = VecGetArray(phi,               &phi_p);            CHKERRXX(ierr);
  ierr = VecGetArray(current_diag,      &current_diag_p);   CHKERRXX(ierr);
  if(desired_diag != NULL)
  {
    ierr = VecGetArrayRead(desired_diag,  &desired_diag_p); CHKERRXX(ierr);
  }

#ifdef P4_TO_P8
  Cube3 cube;
#else
  Cube2 cube;
#endif

  std::vector<p4est_quadrant_t> ngbd;

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);

    for (size_t q=0; q<tree->quadrants.elem_count; ++q){
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      PetscInt quad_gloidx = quad_idx + p4est->global_first_quadrant[p4est->mpirank];

      p4est_locidx_t corners[P4EST_CHILDREN];
      for(int i=0; i<P4EST_CHILDREN; ++i)
        corners[i] = nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];

      double phi_q = phi_cell(quad_idx, phi_p);

      double dtmp = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dx = (xyz_max[0]-xyz_min[0]) * dtmp;
      double dy = (xyz_max[1]-xyz_min[1]) * dtmp;
  #ifdef P4_TO_P8
      double dz = (xyz_max[2]-xyz_min[2]) * dtmp;
  #endif

      double x = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
      double y = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
      double z = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
#endif
      double xyz_q[] = {x, y
                  #ifdef P4_TO_P8
                        , z
                  #endif
                       };

      bool all_pos = true;
      for(int i=0; i<P4EST_CHILDREN; ++i)
        all_pos = all_pos && (phi_p[corners[i]]>0);

      cube.x0 = x - 0.5*dx;
      cube.x1 = x + 0.5*dx;
      cube.y0 = y - 0.5*dy;
      cube.y1 = y + 0.5*dy;

#ifdef P4_TO_P8
      OctValue  p(phi_p[corners[dir::v_mmm]], phi_p[corners[dir::v_mmp]],
                  phi_p[corners[dir::v_mpm]], phi_p[corners[dir::v_mpp]],
                  phi_p[corners[dir::v_pmm]], phi_p[corners[dir::v_pmp]],
                  phi_p[corners[dir::v_ppm]], phi_p[corners[dir::v_ppp]]);

      cube.z0 = z - 0.5*dz;
      cube.z1 = z + 0.5*dz;
      double volume_cut_cell = cube.volume_In_Negative_Domain(p);
#else
      QuadValue p(phi_p[corners[dir::v_mmm]], phi_p[corners[dir::v_mpm]], phi_p[corners[dir::v_pmm]], phi_p[corners[dir::v_ppm]]);
      double volume_cut_cell = cube.area_In_Negative_Domain(p);
#endif

      /* Way inside omega_plus and we dont care! */
      if((bc->interfaceType(xyz_q)==DIRICHLET && phi_q>0) ||
         (bc->interfaceType(xyz_q)==NEUMANN && (all_pos || volume_cut_cell<EPS)))
      {
        ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 1, ADD_VALUES); CHKERRXX(ierr);
        if(!nullspace_use_fixed_point) null_space_p[quad_idx] = 0;
        continue;
      }

      if(!nullspace_use_fixed_point) null_space_p[quad_idx] = 1;

      /* First add the diagonal term */
      if((desired_diag!= NULL) && (desired_diag_p[quad_idx]!=0))
      {
        ierr = MatSetValue(A, quad_gloidx, quad_gloidx, volume_cut_cell*desired_diag_p[quad_idx], ADD_VALUES); CHKERRXX(ierr);
        current_diag_p[quad_idx]  = desired_diag_p[quad_idx];
        matrix_has_nullspace = false;
      }
      else
        current_diag_p[quad_idx]  = 0.0;

      double s;
#ifdef P4_TO_P8
      QuadValue q2;
      Cube2 c2;
#endif

      for(int dir=0; dir<P4EST_FACES; ++dir)
      {
        /* first check if the cell is a wall */
        if(is_quad_Wall(p4est, tree_idx, quad, dir))
        {
          switch(dir)
          {
          case dir::f_m00:
#ifdef P4_TO_P8
            if(bc->wallType(x-.5*dx, y, z)==DIRICHLET)
#else
            if(bc->wallType(x-.5*dx, y)==DIRICHLET)
#endif
            {
              matrix_has_nullspace = false;
#ifdef P4_TO_P8
              c2.x0 = cube.y0; c2.x1 = cube.y1;
              c2.y0 = cube.z0; c2.y1 = cube.z1;
              q2.val00 = p.val000; q2.val01 = p.val001;
              q2.val10 = p.val010; q2.val11 = p.val011;
              s = c2.area_In_Negative_Domain(q2);
#else
              s = dy * fraction_Interval_Covered_By_Irregular_Domain(p.val00, p.val01, dy, dy);
#endif
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu*s/dx, ADD_VALUES); CHKERRXX(ierr);
            }
            break;

          case dir::f_p00:
#ifdef P4_TO_P8
            if(bc->wallType(x+.5*dx, y, z)==DIRICHLET)
#else
            if(bc->wallType(x+.5*dx, y)==DIRICHLET)
#endif
            {
              matrix_has_nullspace = false;
#ifdef P4_TO_P8
              c2.x0 = cube.y0; c2.x1 = cube.y1;
              c2.y0 = cube.z0; c2.y1 = cube.z1;
              q2.val00 = p.val100; q2.val01 = p.val101;
              q2.val10 = p.val110; q2.val11 = p.val111;
              s = c2.area_In_Negative_Domain(q2);
#else
              s = dy * fraction_Interval_Covered_By_Irregular_Domain(p.val10, p.val11, dy, dy);
#endif
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu*s/dx, ADD_VALUES); CHKERRXX(ierr);
            }
            break;

          case dir::f_0m0:
#ifdef P4_TO_P8
            if(bc->wallType(x, y-.5*dy, z)==DIRICHLET)
#else
            if(bc->wallType(x, y-.5*dy)==DIRICHLET)
#endif
            {
              matrix_has_nullspace = false;
#ifdef P4_TO_P8
              c2.x0 = cube.x0; c2.x1 = cube.x1;
              c2.y0 = cube.z0; c2.y1 = cube.z1;
              q2.val00 = p.val000; q2.val01 = p.val001;
              q2.val10 = p.val100; q2.val11 = p.val101;
              s = c2.area_In_Negative_Domain(q2);
#else
              s = dx * fraction_Interval_Covered_By_Irregular_Domain(p.val00, p.val10, dx, dx);
#endif
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu*s/dy, ADD_VALUES); CHKERRXX(ierr);
            }
            break;

          case dir::f_0p0:
#ifdef P4_TO_P8
            if(bc->wallType(x, y+.5*dy, z)==DIRICHLET)
#else
            if(bc->wallType(x, y+.5*dy)==DIRICHLET)
#endif
            {
              matrix_has_nullspace = false;
#ifdef P4_TO_P8
              c2.x0 = cube.x0; c2.x1 = cube.x1;
              c2.y0 = cube.z0; c2.y1 = cube.z1;
              q2.val00 = p.val010; q2.val01 = p.val011;
              q2.val10 = p.val110; q2.val11 = p.val111;
              s = c2.area_In_Negative_Domain(q2);
#else
              s = dx * fraction_Interval_Covered_By_Irregular_Domain(p.val01, p.val11, dx, dx);
#endif
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu*s/dy, ADD_VALUES); CHKERRXX(ierr);
            }
            break;

#ifdef P4_TO_P8
          case dir::f_00m:
            if(bc->wallType(x, y, z-.5*dz)==DIRICHLET)
            {
              matrix_has_nullspace = false;
              c2.x0 = cube.x0; c2.x1 = cube.x1;
              c2.y0 = cube.y0; c2.y1 = cube.y1;
              q2.val00 = p.val000; q2.val01 = p.val010;
              q2.val10 = p.val100; q2.val11 = p.val110;
              s = c2.area_In_Negative_Domain(q2);

              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu*s/dz, ADD_VALUES); CHKERRXX(ierr);
            }
            break;

          case dir::f_00p:
            if(bc->wallType(x, y, z+.5*dz)==DIRICHLET)
            {
              matrix_has_nullspace = false;
              c2.x0 = cube.x0; c2.x1 = cube.x1;
              c2.y0 = cube.y0; c2.y1 = cube.y1;
              q2.val00 = p.val001; q2.val01 = p.val011;
              q2.val10 = p.val101; q2.val11 = p.val111;
              s = c2.area_In_Negative_Domain(q2);

              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, 2*mu*s/dz, ADD_VALUES); CHKERRXX(ierr);
            }
            break;
#endif
          }

          continue;
        }


        /* now get the neighbors */
        ngbd.resize(0);
        ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);

        if(ngbd.size()==1)
        {
          int8_t level_tmp = ngbd[0].level;
          p4est_locidx_t quad_tmp_idx = ngbd[0].p.piggy3.local_num;
          p4est_locidx_t tree_tmp_idx = ngbd[0].p.piggy3.which_tree;

          /* make sure the neighbor is well defined ... important for the nullspace to be correct */
          p4est_locidx_t corners_tmp[P4EST_CHILDREN];
          for(int i=0; i<P4EST_CHILDREN; ++i)
            corners_tmp[i] = nodes->local_nodes[quad_tmp_idx*P4EST_CHILDREN + i];

          double x_tmp = quad_x_fr_q(quad_tmp_idx, tree_tmp_idx, p4est, ghost);
          double y_tmp = quad_y_fr_q(quad_tmp_idx, tree_tmp_idx, p4est, ghost);
#ifdef P4_TO_P8
          double z_tmp = quad_z_fr_q(quad_tmp_idx, tree_tmp_idx, p4est, ghost);
          Cube3 cube_tmp;
#else
          Cube2 cube_tmp;
#endif
          cube_tmp.x0 = x_tmp - 0.5*dx;
          cube_tmp.x1 = x_tmp + 0.5*dx;
          cube_tmp.y0 = y_tmp - 0.5*dy;
          cube_tmp.y1 = y_tmp + 0.5*dy;

#ifdef P4_TO_P8
          OctValue  p_tmp(phi_p[corners_tmp[dir::v_mmm]], phi_p[corners_tmp[dir::v_mmp]],
                          phi_p[corners_tmp[dir::v_mpm]], phi_p[corners_tmp[dir::v_mpp]],
                          phi_p[corners_tmp[dir::v_pmm]], phi_p[corners_tmp[dir::v_pmp]],
                          phi_p[corners_tmp[dir::v_ppm]], phi_p[corners_tmp[dir::v_ppp]]);

          cube_tmp.z0 = z_tmp - 0.5*dz;
          cube_tmp.z1 = z_tmp + 0.5*dz;
          double volume_cut_cell_tmp = cube_tmp.volume_In_Negative_Domain(p_tmp);
#else
          QuadValue p_tmp(phi_p[corners_tmp[dir::v_mmm]], phi_p[corners_tmp[dir::v_mpm]], phi_p[corners_tmp[dir::v_pmm]], phi_p[corners_tmp[dir::v_ppm]]);
          double volume_cut_cell_tmp = cube_tmp.area_In_Negative_Domain(p_tmp);
#endif

          double phi_tmp = phi_cell(quad_tmp_idx, phi_p);

          bool is_pos = false;
          bool is_neg = false;
          switch(dir)
          {
#ifdef P4_TO_P8
          case dir::f_m00:
            is_pos = p.val000>0 || p.val001>0 || p.val010>0 ||p.val011>0;
            is_neg = p.val000<0 || p.val001<0 || p.val010<0 ||p.val011<0; break;
          case dir::f_p00:
            is_pos = p.val100>0 || p.val101>0 || p.val110>0 ||p.val111>0;
            is_neg = p.val100<0 || p.val101<0 || p.val110<0 ||p.val111<0; break;
          case dir::f_0m0:
            is_pos = p.val000>0 || p.val001>0 || p.val100>0 ||p.val101>0;
            is_neg = p.val000<0 || p.val001<0 || p.val100<0 ||p.val101<0; break;
          case dir::f_0p0:
            is_pos = p.val010>0 || p.val011>0 || p.val110>0 ||p.val111>0;
            is_neg = p.val010<0 || p.val011<0 || p.val110<0 ||p.val111<0; break;
          case dir::f_00m:
            is_pos = p.val000>0 || p.val010>0 || p.val100>0 ||p.val110>0;
            is_neg = p.val000<0 || p.val010<0 || p.val100<0 ||p.val110<0; break;
          case dir::f_00p:
            is_pos = p.val001>0 || p.val011>0 || p.val101>0 ||p.val111>0;
            is_neg = p.val001<0 || p.val011<0 || p.val101<0 ||p.val111<0; break;
#else
          case dir::f_m00: is_pos = p.val00>0 || p.val01>0; is_neg = p.val00<0 || p.val01<0; break;
          case dir::f_p00: is_pos = p.val10>0 || p.val11>0; is_neg = p.val10<0 || p.val11<0; break;
          case dir::f_0m0: is_pos = p.val00>0 || p.val10>0; is_neg = p.val00<0 || p.val10<0; break;
          case dir::f_0p0: is_pos = p.val01>0 || p.val11>0; is_neg = p.val01<0 || p.val11<0; break;
#endif
          }

          /* DIRICHLET Boundary Condition */
          if(bc->interfaceType(xyz_q)==DIRICHLET && phi_tmp>0)
          {
            matrix_has_nullspace = false;
            double dtmp;
            switch(dir)
            {
            case dir::f_m00: case dir::f_p00: dtmp = dx; break;
            case dir::f_0m0: case dir::f_0p0: dtmp = dy; break;
#ifdef P4_TO_P8
            case dir::f_00m: case dir::f_00p: dtmp = dz; break;
#endif
            default: throw std::invalid_argument("[ERROR]: unknown direction.");
            }

            double theta = fraction_Interval_Covered_By_Irregular_Domain(phi_q, phi_tmp, dtmp, dtmp);
            if(theta<EPS) theta = EPS;
            if(theta>1  ) theta = 1;
            switch(dir)
            {
#ifdef P4_TO_P8
            case dir::f_m00:
            case dir::f_p00:
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, mu/theta * dy*dz/dx, ADD_VALUES); CHKERRXX(ierr);
              break;
            case dir::f_0m0:
            case dir::f_0p0:
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, mu/theta * dx*dz/dy, ADD_VALUES); CHKERRXX(ierr);
              break;
            case dir::f_00m:
            case dir::f_00p:
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, mu/theta * dx*dy/dz, ADD_VALUES); CHKERRXX(ierr);
              break;
#else
            case dir::f_m00:
            case dir::f_p00:
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, mu/theta * dy/dx, ADD_VALUES); CHKERRXX(ierr);
              break;
            case dir::f_0m0:
            case dir::f_0p0:
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx, mu/theta * dx/dy, ADD_VALUES); CHKERRXX(ierr);
              break;
#endif
            }
          }
          /* NEUMANN Boundary Condition */
          else if(bc->interfaceType(xyz_q)==NEUMANN && is_pos && is_neg && volume_cut_cell_tmp>EPS)
          {
            double d;
            switch(dir)
            {
#ifdef P4_TO_P8
            case dir::f_m00:
              c2.x0 = cube.y0; c2.x1 = cube.y1; c2.y0 = cube.z0; c2.y1 = cube.z1;
              q2.val00 = p.val000; q2.val01 = p.val001; q2.val10 = p.val010; q2.val11 = p.val011;
              s = c2.area_In_Negative_Domain(q2); d = dx;
              break;
            case dir::f_p00:
              c2.x0 = cube.y0; c2.x1 = cube.y1; c2.y0 = cube.z0; c2.y1 = cube.z1;
              q2.val00 = p.val100; q2.val01 = p.val101; q2.val10 = p.val110; q2.val11 = p.val111;
              s = c2.area_In_Negative_Domain(q2); d = dx;
              break;
            case dir::f_0m0:
              c2.x0 = cube.x0; c2.x1 = cube.x1; c2.y0 = cube.z0; c2.y1 = cube.z1;
              q2.val00 = p.val000; q2.val01 = p.val001; q2.val10 = p.val100; q2.val11 = p.val101;
              s = c2.area_In_Negative_Domain(q2); d = dy;
              break;
            case dir::f_0p0:
              c2.x0 = cube.x0; c2.x1 = cube.x1; c2.y0 = cube.z0; c2.y1 = cube.z1;
              q2.val00 = p.val010; q2.val01 = p.val011; q2.val10 = p.val110; q2.val11 = p.val111;
              s = c2.area_In_Negative_Domain(q2); d = dy;
              break;
            case dir::f_00m:
              c2.x0 = cube.x0; c2.x1 = cube.x1; c2.y0 = cube.y0; c2.y1 = cube.y1;
              q2.val00 = p.val000; q2.val01 = p.val010; q2.val10 = p.val100; q2.val11 = p.val110;
              s = c2.area_In_Negative_Domain(q2); d = dz;
              break;
            case dir::f_00p:
              c2.x0 = cube.x0; c2.x1 = cube.x1; c2.y0 = cube.y0; c2.y1 = cube.y1;
              q2.val00 = p.val001; q2.val01 = p.val011; q2.val10 = p.val101; q2.val11 = p.val111;
              s = c2.area_In_Negative_Domain(q2); d = dz;
              break;
#else
            case dir::f_m00: s = dy*fraction_Interval_Covered_By_Irregular_Domain(p.val00, p.val01, dy, dy); d = dx; break;
            case dir::f_p00: s = dy*fraction_Interval_Covered_By_Irregular_Domain(p.val10, p.val11, dy, dy); d = dx; break;
            case dir::f_0m0: s = dx*fraction_Interval_Covered_By_Irregular_Domain(p.val00, p.val10, dx, dx); d = dy; break;
            case dir::f_0p0: s = dx*fraction_Interval_Covered_By_Irregular_Domain(p.val01, p.val11, dx, dx); d = dy; break;
#endif
            }

            if(s>EPS)
            {
              ierr = MatSetValue(A, quad_gloidx, quad_gloidx                       ,  mu*s/d, ADD_VALUES); CHKERRXX(ierr);
              ierr = MatSetValue(A, quad_gloidx, compute_global_index(quad_tmp_idx), -mu*s/d, ADD_VALUES); CHKERRXX(ierr);
            }
          }
          /* no interface - regular discretization */
          else if(is_neg && !(bc->interfaceType(xyz_q)==NEUMANN && volume_cut_cell_tmp<EPS))
          {
            double s_tmp = pow((double)P4EST_QUADRANT_LEN(ngbd[0].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);

            ngbd.resize(0);
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_tmp_idx, tree_tmp_idx, dir%2==0 ? dir+1 : dir-1);

            std::vector<double> s_ng(ngbd.size());
            for(unsigned int i=0; i<ngbd.size(); ++i)
              s_ng[i] = pow((double)P4EST_QUADRANT_LEN(ngbd[i].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);

            double d = 0;
            for(unsigned int i=0; i<ngbd.size(); ++i)
              d += s_ng[i]/s_tmp * 0.5 * (double)(P4EST_QUADRANT_LEN(level_tmp)+P4EST_QUADRANT_LEN(ngbd[i].level))/(double)P4EST_ROOT_LEN;

            d *= (xyz_max[dir/2]-xyz_min[dir/2]);

            switch(dir)
            {
#ifdef P4_TO_P8
            case dir::f_m00: case dir::f_p00: s = dy*dz; break;
            case dir::f_0m0: case dir::f_0p0: s = dx*dz; break;
            case dir::f_00m: case dir::f_00p: s = dx*dy; break;
#else
            case dir::f_m00: case dir::f_p00: s = dy; break;
            case dir::f_0m0: case dir::f_0p0: s = dx; break;
#endif
            default: throw std::invalid_argument("[ERROR]: unknown direction.");
            }

            for(unsigned int i=0; i<ngbd.size(); ++i)
            {
              ierr = MatSetValue(A, quad_gloidx, compute_global_index(ngbd[i].p.piggy3.local_num), mu*s * s_ng[i]/s_tmp/d, ADD_VALUES); CHKERRXX(ierr);
            }

            ierr = MatSetValue(A, quad_gloidx, compute_global_index(quad_tmp_idx), -mu*s/d, ADD_VALUES); CHKERRXX(ierr);

            if(nullspace_use_fixed_point && quad_gloidx<fixed_value_idx_g)
            {
              fixed_value_idx_l = quad_idx;
              fixed_value_idx_g = quad_gloidx;
            }
          }
        }
        /* there is more than one neighbor, regular bulk case. This assumes uniform on interface ! */
        else if(ngbd.size()>1)
        {
          double s_tmp = pow((double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);

          std::vector<double> s_ng(ngbd.size());
          for(unsigned int i=0; i<ngbd.size(); ++i)
            s_ng[i] = pow((double)P4EST_QUADRANT_LEN(ngbd[i].level)/(double)P4EST_ROOT_LEN, (double)P4EST_DIM-1);

          double d = 0;
          for(unsigned int i=0; i<ngbd.size(); ++i)
            d += s_ng[i]/s_tmp * 0.5 * (double)(P4EST_QUADRANT_LEN(quad->level)+P4EST_QUADRANT_LEN(ngbd[i].level))/(double)P4EST_ROOT_LEN;

          d *= (xyz_max[dir/2]-xyz_min[dir/2]);

          switch(dir)
          {
#ifdef P4_TO_P8
          case dir::f_m00: case dir::f_p00: s = dy*dz; break;
          case dir::f_0m0: case dir::f_0p0: s = dx*dz; break;
          case dir::f_00m: case dir::f_00p: s = dx*dy; break;
#else
          case dir::f_m00: case dir::f_p00: s = dy; break;
          case dir::f_0m0: case dir::f_0p0: s = dx; break;
#endif
          default: throw std::invalid_argument("[ERROR]: unknown direction.");
          }

          for(unsigned int i=0; i<ngbd.size(); ++i)
          {
            ierr = MatSetValue(A, quad_gloidx, compute_global_index(ngbd[i].p.piggy3.local_num), -mu*s * s_ng[i]/s_tmp/d, ADD_VALUES); CHKERRXX(ierr);
          }

          ierr = MatSetValue(A, quad_gloidx, quad_gloidx, mu*s/d, ADD_VALUES); CHKERRXX(ierr);

          if(nullspace_use_fixed_point && quad_gloidx<fixed_value_idx_g)
          {
            fixed_value_idx_l = quad_idx;
            fixed_value_idx_g = quad_gloidx;
          }
        }
      }
    }
  }

  if(!nullspace_use_fixed_point)
  {
    ierr = VecRestoreArray(null_space, &null_space_p); CHKERRXX(ierr);
  }

  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

  // restore pointers
  ierr = VecRestoreArray(phi,               &phi_p          );    CHKERRXX(ierr);
  ierr = VecRestoreArray(current_diag,      &current_diag_p );    CHKERRXX(ierr);
  if(desired_diag != NULL)
  {
    ierr = VecRestoreArrayRead(desired_diag,  &desired_diag_p );  CHKERRXX(ierr);
  }

//  PetscViewer view;
//  char name[1000];
//  sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/mat_p4est.m");
//  ierr = PetscViewerASCIIOpen(p4est->mpicomm, name, &view); CHKERRXX(ierr);
//  ierr = PetscViewerSetFormat(view, PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(ierr);
//  ierr = MatView(A, view); CHKERRXX(ierr);

//  sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/rhs_p4est.m");
//  ierr = PetscViewerASCIIOpen(p4est->mpicomm, name, &view); CHKERRXX(ierr);
//  ierr = PetscViewerSetFormat(view, PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(ierr);
//  ierr = VecView(rhs, view); CHKERRXX(ierr);

  // check for null space
  ierr = MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace, 1, MPI_INT, MPI_LAND, p4est->mpicomm); CHKERRXX(ierr);
  if (matrix_has_nullspace)
  {
    if(!nullspace_use_fixed_point)
    {
      if(A_null_space != NULL)
      {
        ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr);
      }

      ierr = VecGhostUpdateBegin(null_space, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (null_space, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      double norm;
      ierr = VecNormalize(null_space, &norm); CHKERRXX(ierr);
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_FALSE, 1, &null_space, &A_null_space); CHKERRXX(ierr);

      ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
      ierr = MatSetTransposeNullSpace(A, A_null_space); CHKERRXX(ierr);
    } else {
      ierr = MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE); CHKERRXX(ierr);
      p4est_gloidx_t fixed_value_idx;
      MPI_Allreduce(&fixed_value_idx_g, &fixed_value_idx, 1, MPI_LONG_LONG_INT, MPI_MIN, p4est->mpicomm);
      if(fixed_value_idx_g>=p4est->global_num_quadrants)
        throw std::invalid_argument("my_p4est_poisson_cells_t->setup_negative_laplace_matrix: could not fix value for all neumann problem. Maybe there is no point inside the domain and away from the interface?");
      if (fixed_value_idx_g != fixed_value_idx){ // we are not setting the fixed value
        fixed_value_idx_l = -1;
        fixed_value_idx_g = fixed_value_idx;
        ierr = MatZeroRows(A, 0, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
      } else {
      // reset the value
        ierr = MatZeroRows(A, 1, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
      }
    }
  }

  if(!nullspace_use_fixed_point)
  {
    ierr = VecDestroy(null_space); CHKERRXX(ierr);
    null_space = NULL;
  }
  is_matrix_ready = true;

  ierr = PetscLogEventEnd(log_my_p4est_poisson_cells_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

//  PetscViewer view;
//  char name[1000];
//  sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/mat_%d.m", p4est->mpisize);
//  ierr = PetscViewerASCIIOpen(p4est->mpicomm, name, &view); CHKERRXX(ierr);
//  ierr = PetscViewerSetFormat(view, PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(ierr);
//  ierr = MatView(A, view); CHKERRXX(ierr);
//  MatView(A, PETSC_VIEWER_STDOUT_WORLD);
}

void my_p4est_poisson_cells_t::update_matrix_diag_only()
{
  P4EST_ASSERT(!is_matrix_ready && only_diag_is_modified);

  matrix_has_nullspace = true;
  double *null_space_p;
  if(!nullspace_use_fixed_point)
  {
    if(null_space != NULL)
    {
      ierr = VecDestroy(null_space); CHKERRXX(ierr);
    }
    ierr = VecDuplicate(rhs, &null_space); CHKERRXX(ierr);
    ierr = VecGetArray(null_space, &null_space_p); CHKERRXX(ierr);
  }

  fixed_value_idx_g = p4est->global_num_quadrants;

  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_cells_update_matrix_diag_only, A, 0, 0, 0); CHKERRXX(ierr);
  double *phi_p, *current_diag_p;
  const double* desired_diag_p;

  ierr = VecGetArray(phi,               &phi_p);            CHKERRXX(ierr);
  ierr = VecGetArray(current_diag,      &current_diag_p);   CHKERRXX(ierr);
  if(desired_diag != NULL)
  {
    ierr = VecGetArrayRead(desired_diag,  &desired_diag_p); CHKERRXX(ierr);
  }

#ifdef P4_TO_P8
  Cube3 cube;
#else
  Cube2 cube;
#endif

  std::vector<p4est_quadrant_t> ngbd;

  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);

    for (size_t q=0; q<tree->quadrants.elem_count; ++q){
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;
      PetscInt quad_gloidx = quad_idx + p4est->global_first_quadrant[p4est->mpirank];

      p4est_locidx_t corners[P4EST_CHILDREN];
      for(int i=0; i<P4EST_CHILDREN; ++i)
        corners[i] = nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];

      double phi_q = phi_cell(quad_idx, phi_p);

      double dtmp = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dx = (xyz_max[0]-xyz_min[0]) * dtmp;
      double dy = (xyz_max[1]-xyz_min[1]) * dtmp;
  #ifdef P4_TO_P8
      double dz = (xyz_max[2]-xyz_min[2]) * dtmp;
  #endif

      double x = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
      double y = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
      double z = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
#endif
      double xyz_q[] = {x, y
                  #ifdef P4_TO_P8
                        , z
                  #endif
                       };

      bool all_pos = true;
      for(int i=0; i<P4EST_CHILDREN; ++i)
        all_pos = all_pos && (phi_p[corners[i]]>0);

      cube.x0 = x - 0.5*dx;
      cube.x1 = x + 0.5*dx;
      cube.y0 = y - 0.5*dy;
      cube.y1 = y + 0.5*dy;

#ifdef P4_TO_P8
      OctValue  p(phi_p[corners[dir::v_mmm]], phi_p[corners[dir::v_mmp]],
                  phi_p[corners[dir::v_mpm]], phi_p[corners[dir::v_mpp]],
                  phi_p[corners[dir::v_pmm]], phi_p[corners[dir::v_pmp]],
                  phi_p[corners[dir::v_ppm]], phi_p[corners[dir::v_ppp]]);

      cube.z0 = z - 0.5*dz;
      cube.z1 = z + 0.5*dz;
      double volume_cut_cell = cube.volume_In_Negative_Domain(p);
#else
      QuadValue p(phi_p[corners[dir::v_mmm]], phi_p[corners[dir::v_mpm]], phi_p[corners[dir::v_pmm]], phi_p[corners[dir::v_ppm]]);
      double volume_cut_cell = cube.area_In_Negative_Domain(p);
#endif

      /* Way inside omega_plus and we dont care! */
      if((bc->interfaceType(xyz_q)==DIRICHLET && phi_q>0) ||
         (bc->interfaceType(xyz_q)==NEUMANN && (all_pos || volume_cut_cell<EPS)))
      {
        if(!nullspace_use_fixed_point)
          null_space_p[quad_idx] = 0;
        continue;
      }

      if(!nullspace_use_fixed_point)
        null_space_p[quad_idx] = 1;

      /* First add the diagonal term */
      ierr = MatSetValue(A, quad_gloidx, quad_gloidx, volume_cut_cell*(((desired_diag != NULL)? desired_diag_p[quad_idx] : 0.0) - current_diag_p[quad_idx]), ADD_VALUES); CHKERRXX(ierr);
      current_diag_p[quad_idx]  = ((desired_diag != NULL)? desired_diag_p[quad_idx] : 0.0);
      if((desired_diag != NULL) && (current_diag_p[quad_idx]!=0))
        matrix_has_nullspace = false;
    }
  }

  if(!nullspace_use_fixed_point)
  {
    ierr = VecRestoreArray(null_space, &null_space_p); CHKERRXX(ierr);
  }

  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

  // restore pointers
  ierr = VecRestoreArray(phi,               &phi_p          );    CHKERRXX(ierr);
  ierr = VecRestoreArray(current_diag,      &current_diag_p );    CHKERRXX(ierr);
  if(desired_diag != NULL)
  {
    ierr = VecRestoreArrayRead(desired_diag,  &desired_diag_p );  CHKERRXX(ierr);
  }

  // check for null space
  ierr = MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace, 1, MPI_INT, MPI_LAND, p4est->mpicomm); CHKERRXX(ierr);
  if (matrix_has_nullspace)
  {
    if(!nullspace_use_fixed_point)
    {
      if(A_null_space != NULL)
      {
        ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr);
      }

      ierr = VecGhostUpdateBegin(null_space, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
      ierr = VecGhostUpdateEnd  (null_space, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

      double norm;
      ierr = VecNormalize(null_space, &norm); CHKERRXX(ierr);
      ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_FALSE, 1, &null_space, &A_null_space); CHKERRXX(ierr);

      ierr = MatSetNullSpace(A, A_null_space); CHKERRXX(ierr);
      ierr = MatSetTransposeNullSpace(A, A_null_space); CHKERRXX(ierr);
    } else {
      ierr = MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE); CHKERRXX(ierr);
      p4est_gloidx_t fixed_value_idx;
      MPI_Allreduce(&fixed_value_idx_g, &fixed_value_idx, 1, MPI_LONG_LONG_INT, MPI_MIN, p4est->mpicomm);
      if(fixed_value_idx_g>=p4est->global_num_quadrants)
        throw std::invalid_argument("my_p4est_poisson_cells_t->setup_negative_laplace_matrix: could not fix value for all neumann problem. Maybe there is no point inside the domain and away from the interface?");
      if (fixed_value_idx_g != fixed_value_idx){ // we are not setting the fixed value
        fixed_value_idx_l = -1;
        fixed_value_idx_g = fixed_value_idx;
        ierr = MatZeroRows(A, 0, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
      } else {
      // reset the value
        ierr = MatZeroRows(A, 1, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
      }
    }
  }

  if(!nullspace_use_fixed_point)
  {
    ierr = VecDestroy(null_space); CHKERRXX(ierr);
    null_space = NULL;
  }

  is_matrix_ready = true;

  ierr = PetscLogEventEnd(log_my_p4est_poisson_cells_update_matrix_diag_only, A, 0, 0, 0); CHKERRXX(ierr);
}



void my_p4est_poisson_cells_t::setup_negative_laplace_rhsvec()
{
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_cells_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

  double *phi_p, *rhs_p;
  ierr = VecGetArray(phi,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(rhs,    &rhs_p   ); CHKERRXX(ierr);

#ifdef P4_TO_P8
  Cube3 cube;
#else
  Cube2 cube;
#endif

  std::vector<p4est_quadrant_t> ngbd;

  /* Main loop over all local quadrant */
  for(p4est_topidx_t tree_idx = p4est->first_local_tree; tree_idx <= p4est->last_local_tree; ++tree_idx){
    p4est_tree_t *tree = (p4est_tree_t*)sc_array_index(p4est->trees, tree_idx);

    for (size_t q=0; q<tree->quadrants.elem_count; ++q){
      const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&tree->quadrants, q);
      p4est_locidx_t quad_idx = q + tree->quadrants_offset;

      p4est_locidx_t corners[P4EST_CHILDREN];
      for(int i=0; i<P4EST_CHILDREN; ++i)
        corners[i] = nodes->local_nodes[quad_idx*P4EST_CHILDREN + i];

      double phi_q = phi_cell(quad_idx, phi_p);

      double dtmp = (double)P4EST_QUADRANT_LEN(quad->level)/(double)P4EST_ROOT_LEN;
      double dx = (xyz_max[0]-xyz_min[0]) * dtmp;
      double dy = (xyz_max[1]-xyz_min[1]) * dtmp;
  #ifdef P4_TO_P8
      double dz = (xyz_max[2]-xyz_min[2]) * dtmp;
  #endif

      double x = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
      double y = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
      double z = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
#endif
      double xyz_q[] = {x, y
                       #ifdef P4_TO_P8
                        , z
                       #endif
                       };

      bool all_pos = true;
      bool is_pos = false;
      for(int i=0; i<P4EST_CHILDREN; ++i)
      {
        all_pos = all_pos && (phi_p[corners[i]]>0);
        is_pos = is_pos || (phi_p[corners[i]]>0);
      }

      cube.x0 = x - 0.5*dx;
      cube.x1 = x + 0.5*dx;
      cube.y0 = y - 0.5*dy;
      cube.y1 = y + 0.5*dy;

#ifdef P4_TO_P8
      OctValue  p(phi_p[corners[dir::v_mmm]], phi_p[corners[dir::v_mmp]],
                  phi_p[corners[dir::v_mpm]], phi_p[corners[dir::v_mpp]],
                  phi_p[corners[dir::v_pmm]], phi_p[corners[dir::v_pmp]],
                  phi_p[corners[dir::v_ppm]], phi_p[corners[dir::v_ppp]]);

      cube.z0 = z - 0.5*dz;
      cube.z1 = z + 0.5*dz;
      double volume_cut_cell = cube.volume_In_Negative_Domain(p);
#else
      QuadValue p(phi_p[corners[dir::v_mmm]], phi_p[corners[dir::v_mpm]], phi_p[corners[dir::v_pmm]], phi_p[corners[dir::v_ppm]]);
      double volume_cut_cell = cube.area_In_Negative_Domain(p);
#endif

      /* Way inside omega_plus and we dont care! */
      if((bc->interfaceType(xyz_q)==DIRICHLET && phi_q>0) ||
         (bc->interfaceType(xyz_q)==NEUMANN && (all_pos || volume_cut_cell<EPS)))
      {
        rhs_p[quad_idx] = 0;
        continue;
      }

      rhs_p[quad_idx] *= volume_cut_cell;

      /* Neumann BC */
      if( bc->interfaceType(xyz_q)==NEUMANN && is_pos )
      {
#ifdef P4_TO_P8
        OctValue interface_values;
        interface_values.val000 = bc->interfaceValue(cube.x0, cube.y0, cube.z0);
        interface_values.val001 = bc->interfaceValue(cube.x0, cube.y0, cube.z1);
        interface_values.val010 = bc->interfaceValue(cube.x0, cube.y1, cube.z0);
        interface_values.val011 = bc->interfaceValue(cube.x0, cube.y1, cube.z1);
        interface_values.val100 = bc->interfaceValue(cube.x1, cube.y0, cube.z0);
        interface_values.val101 = bc->interfaceValue(cube.x1, cube.y0, cube.z1);
        interface_values.val110 = bc->interfaceValue(cube.x1, cube.y1, cube.z0);
        interface_values.val111 = bc->interfaceValue(cube.x1, cube.y1, cube.z1);
#else
        QuadValue interface_values;
        interface_values.val00 = bc->interfaceValue(cube.x0, cube.y0);
        interface_values.val01 = bc->interfaceValue(cube.x0, cube.y1);
        interface_values.val10 = bc->interfaceValue(cube.x1, cube.y0);
        interface_values.val11 = bc->interfaceValue(cube.x1, cube.y1);
#endif

        double val_interface = cube.integrate_Over_Interface(interface_values,p);
        rhs_p[quad_idx] += mu*val_interface;
      }

      /* Dirichlet BC */
      if(bc->interfaceType(xyz_q)==DIRICHLET && fabs(phi_q)<=diag_min)
      {
        for(int dir=0; dir<P4EST_FACES; ++dir)
        {
          if(!is_quad_Wall(p4est, tree_idx, quad, dir))
          {
            ngbd.resize(0);
            ngbd_c->find_neighbor_cells_of_cell(ngbd, quad_idx, tree_idx, dir);

            double phi_tmp = phi_cell(ngbd[0].p.piggy3.local_num, phi_p);

            if(phi_tmp*phi_q < 0)
            {
              double dtmp;
              switch(dir)
              {
              case dir::f_m00: case dir::f_p00: dtmp = dx; break;
              case dir::f_0m0: case dir::f_0p0: dtmp = dy; break;
#ifdef P4_TO_P8
              case dir::f_00m: case dir::f_00p: dtmp = dz; break;
#endif
              default: throw std::invalid_argument("[ERROR]: unknown direction.");
              }

              double theta = fraction_Interval_Covered_By_Irregular_Domain(phi_q, phi_tmp, dtmp, dtmp);
              if (theta < EPS) theta = EPS;
              if (theta > 1  ) theta = 1;

              switch(dir)
              {
#ifdef P4_TO_P8
              case dir::f_m00: rhs_p[quad_idx] += mu*dy*dz/dx/theta * bc->interfaceValue(x-theta*dx, y, z); break;
              case dir::f_p00: rhs_p[quad_idx] += mu*dy*dz/dx/theta * bc->interfaceValue(x+theta*dx, y, z); break;
              case dir::f_0m0: rhs_p[quad_idx] += mu*dx*dz/dy/theta * bc->interfaceValue(x, y-theta*dy, z); break;
              case dir::f_0p0: rhs_p[quad_idx] += mu*dx*dz/dy/theta * bc->interfaceValue(x, y+theta*dy, z); break;
              case dir::f_00m: rhs_p[quad_idx] += mu*dx*dy/dz/theta * bc->interfaceValue(x, y, z-theta*dz); break;
              case dir::f_00p: rhs_p[quad_idx] += mu*dx*dy/dz/theta * bc->interfaceValue(x, y, z+theta*dz); break;
#else
              case dir::f_m00: rhs_p[quad_idx] += mu*dy/dx/theta * bc->interfaceValue(x-theta*dx, y); break;
              case dir::f_p00: rhs_p[quad_idx] += mu*dy/dx/theta * bc->interfaceValue(x+theta*dx, y); break;
              case dir::f_0m0: rhs_p[quad_idx] += mu*dx/dy/theta * bc->interfaceValue(x, y-theta*dy); break;
              case dir::f_0p0: rhs_p[quad_idx] += mu*dx/dy/theta * bc->interfaceValue(x, y+theta*dy); break;
#endif
              }
            }
          }
        }
      }

#ifdef P4_TO_P8
      Cube2 c2;
      QuadValue qval;
#endif
      double val_wall;
      double s;
      double d;
      BoundaryConditionType bc_wtype;

      /* accounting for the contributions from the wall */
      for(int dir=0; dir<P4EST_FACES; ++dir)
      {
        if(is_quad_Wall(p4est, tree_idx, quad, dir))
        {
          switch(dir)
          {
#ifdef P4_TO_P8
          case dir::f_m00:
            bc_wtype = bc->wallType (x-.5*dx, y, z);
            val_wall = bc->wallValue(x-.5*dx, y, z);
            c2.x0 = cube.y0; c2.x1 = cube.y1;
            c2.y0 = cube.z0; c2.y1 = cube.z1;
            qval.val00 = p.val000; qval.val01 = p.val001;
            qval.val10 = p.val010; qval.val11 = p.val011;
            s = c2.area_In_Negative_Domain(qval);
            d = dx;
            break;
          case dir::f_p00:
            bc_wtype = bc->wallType (x+.5*dx, y, z);
            val_wall = bc->wallValue(x+.5*dx, y, z);
            c2.x0 = cube.y0; c2.x1 = cube.y1;
            c2.y0 = cube.z0; c2.y1 = cube.z1;
            qval.val00 = p.val100; qval.val01 = p.val101;
            qval.val10 = p.val110; qval.val11 = p.val111;
            s = c2.area_In_Negative_Domain(qval);
            d = dx;
            break;
          case dir::f_0m0:
            bc_wtype = bc->wallType (x, y-.5*dy, z);
            val_wall = bc->wallValue(x, y-.5*dy, z);
            c2.x0 = cube.x0; c2.x1 = cube.x1;
            c2.y0 = cube.z0; c2.y1 = cube.z1;
            qval.val00 = p.val000; qval.val01 = p.val001;
            qval.val10 = p.val100; qval.val11 = p.val101;
            s = c2.area_In_Negative_Domain(qval);
            d = dy;
            break;
          case dir::f_0p0:
            bc_wtype = bc->wallType (x, y+.5*dy, z);
            val_wall = bc->wallValue(x, y+.5*dy, z);
            c2.x0 = cube.x0; c2.x1 = cube.x1;
            c2.y0 = cube.z0; c2.y1 = cube.z1;
            qval.val00 = p.val010; qval.val01 = p.val011;
            qval.val10 = p.val110; qval.val11 = p.val111;
            s = c2.area_In_Negative_Domain(qval);
            d = dy;
            break;
          case dir::f_00m:
            bc_wtype = bc->wallType (x, y, z-.5*dz);
            val_wall = bc->wallValue(x, y, z-.5*dz);
            c2.x0 = cube.x0; c2.x1 = cube.x1;
            c2.y0 = cube.y0; c2.y1 = cube.y1;
            qval.val00 = p.val000; qval.val01 = p.val010;
            qval.val10 = p.val100; qval.val11 = p.val110;
            s = c2.area_In_Negative_Domain(qval);
            d = dz;
            break;
          case dir::f_00p:
            bc_wtype = bc->wallType (x, y, z+.5*dz);
            val_wall = bc->wallValue(x, y, z+.5*dz);
            c2.x0 = cube.x0; c2.x1 = cube.x1;
            c2.y0 = cube.y0; c2.y1 = cube.y1;
            qval.val00 = p.val001; qval.val01 = p.val011;
            qval.val10 = p.val101; qval.val11 = p.val111;
            s = c2.area_In_Negative_Domain(qval);
            d = dz;
            break;
#else
          case dir::f_m00:
            bc_wtype = bc->wallType (x-.5*dx, y);
            val_wall = bc->wallValue(x-.5*dx, y);
            s = dy*fraction_Interval_Covered_By_Irregular_Domain(p.val00, p.val01, dx, dx);
            d = dx;
            break;
          case dir::f_p00:
            bc_wtype = bc->wallType (x+.5*dx, y);
            val_wall = bc->wallValue(x+.5*dx, y);
            s = dy*fraction_Interval_Covered_By_Irregular_Domain(p.val10, p.val11, dx, dx);
            d = dx;
            break;
          case dir::f_0m0:
            bc_wtype = bc->wallType (x, y-.5*dy);
            val_wall = bc->wallValue(x, y-.5*dy);
            s = dx*fraction_Interval_Covered_By_Irregular_Domain(p.val00, p.val10, dy, dy);
            d = dy;
            break;
          case dir::f_0p0:
            bc_wtype = bc->wallType (x, y+.5*dy);
            val_wall = bc->wallValue(x, y+.5*dy);
            s = dx*fraction_Interval_Covered_By_Irregular_Domain(p.val01, p.val11, dy, dy);
            d = dy;
            break;
#endif
          }

          switch(bc_wtype)
          {
          case DIRICHLET:
            rhs_p[quad_idx] += 2*mu*val_wall/d * s;
            break;
          case NEUMANN:
            rhs_p[quad_idx] += mu*val_wall * s;
            break;
          default:
            throw std::invalid_argument("[ERROR]: my_p4est_poisson_cell_base->setup_negative_laplace_rhsvec: unknown boundary condition.");
          }
        }
      }
    }
  }

//  PetscViewer view;
//  char name[1000];
//  sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/rhs_p4est.m");
//  ierr = PetscViewerASCIIOpen(p4est->mpicomm, name, &view); CHKERRXX(ierr);
//  ierr = PetscViewerSetFormat(view, PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(ierr);
//  ierr = VecView(rhs, view); CHKERRXX(ierr);

  if (matrix_has_nullspace)
  {
    if(!nullspace_use_fixed_point)
    {
      ierr = MatNullSpaceRemove(A_null_space, rhs, NULL); CHKERRXX(ierr);
    }
    else if(fixed_value_idx_l >= 0)
      rhs_p[fixed_value_idx_l] = 0;
  }

  // restore the pointers
  ierr = VecRestoreArray(phi,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs,    &rhs_p   ); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_cells_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);

//  PetscViewer view;
//  char name[1000];
//  sprintf(name, "/home/guittet/code/Output/p4est_navier_stokes/rhs_%d.m", p4est->mpisize);
//  ierr = PetscViewerASCIIOpen(p4est->mpicomm, name, &view); CHKERRXX(ierr);
//  ierr = PetscViewerSetFormat(view, PETSC_VIEWER_ASCII_MATLAB); CHKERRXX(ierr);
//  ierr = VecView(rhs, view); CHKERRXX(ierr);
//  PetscPrintf(p4est->mpicomm, "SAVED RHS\n");
}

