#ifdef P4_TO_P8
#include "my_p8est_poisson_node_base.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
#else
#include "my_p4est_poisson_node_base.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/cube2.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/CASL_math.h>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_PoissonSolverNodeBase_matrix_preallocation;
extern PetscLogEvent log_PoissonSolverNodeBase_matrix_setup;
extern PetscLogEvent log_PoissonSolverNodeBase_rhsvec_setup;
extern PetscLogEvent log_PoissonSolverNodeBase_solve;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bc_strength 1.0

PoissonSolverNodeBase::PoissonSolverNodeBase(const my_p4est_node_neighbors_t *node_neighbors)
  : node_neighbors_(node_neighbors),
    p4est(node_neighbors->p4est), nodes(node_neighbors->nodes), ghost(node_neighbors->ghost), myb_(node_neighbors->myb),
    phi_interp(p4est, nodes, ghost, myb_, node_neighbors),
    phi_cf(NULL),
    mu_(1.), diag_add_(0.),
    is_matrix_computed(false), matrix_has_nullspace(false),
    bc_(NULL), A(NULL),
    is_phi_dd_owned(false), is_mue_dd_owned(false),
    rhs_(NULL), phi_(NULL), add_(NULL), mue_(NULL),
    phi_xx_(NULL), phi_yy_(NULL), mue_xx_(NULL), mue_yy_(NULL), robin_coef_(NULL)
  #ifdef P4_TO_P8
    ,
    phi_zz_(NULL), mue_zz_(NULL)
  #endif
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
  ierr = KSPSetTolerances(ksp, 1e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);

  splitting_criteria_t *data = (splitting_criteria_t*)p4est->user_pointer;

  // compute grid parameters
  // NOTE: Assuming all trees are of the same size [0, 1]^d
  dx_min = 1.0 / pow(2.,(double) data->max_lvl);
  dy_min = dx_min;
#ifdef P4_TO_P8
  dz_min = dx_min;
#endif
#ifdef P4_TO_P8
  d_min = MIN(dx_min, dy_min, dz_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min + dz_min*dz_min);
#else
  d_min = MIN(dx_min, dy_min);
  diag_min = sqrt(dx_min*dx_min + dy_min*dy_min);
#endif

  // construct petsc global indices
  petsc_gloidx.resize(nodes->indep_nodes.elem_count);

  // local nodes
  for (p4est_locidx_t i = 0; i<nodes->num_owned_indeps; i++)
    petsc_gloidx[i] = global_node_offset[p4est->mpirank] + i;

  // ghost nodes
  p4est_locidx_t ghost_size = nodes->indep_nodes.elem_count - nodes->num_owned_indeps;
  for (p4est_locidx_t i = 0; i<ghost_size; i++){
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, i + nodes->num_owned_indeps);
    petsc_gloidx[i+nodes->num_owned_indeps] = global_node_offset[nodes->nonlocal_ranks[i]] + ni->p.piggy3.local_num;
  }

  /* We handle the all-neumann situation by fixing the solution at some point inside the domain.
   * In general this method is _NOT_ recommended as it pollutes the eigen-value spectrum.
   *
   * The other (and preferred) method to solve siongular matrices is to set the nullspace. If the
   * matrix is non-symmetric (which is the case for us) one has to compute the left nullspace, that
   * is the nuullspace of A^T instead of A. This is because it is the left nullspace that is orthogonal
   * complement of Range(A). Unfortunately teh general way of computing the nullspace is the SVD algorithm
   * which is more expensive than the linear system solution itself! Unless you can come up with a
   * smart way to "guess" the left nullspace, this is not recommened.
   *
   * To fix the solution, every processor sets up the matrix as if it was non-singular. Next, if all
   * processes agree that the matrix is singular, the process with the first interior node fixes the
   * value at that point to zero. This is done by modifing the row corresponding to 'fixed_value_idx_g'
   * index in the matrix.
   */
  fixed_value_idx_g = global_node_offset[p4est->mpisize];
}

PoissonSolverNodeBase::~PoissonSolverNodeBase()
{
  if (A             != NULL) ierr = MatDestroy(A);                      CHKERRXX(ierr);
  if (ksp           != NULL) ierr = KSPDestroy(ksp);                    CHKERRXX(ierr);
  if (is_phi_dd_owned){
    if (phi_xx_     != NULL) ierr = VecDestroy(phi_xx_);                CHKERRXX(ierr);
    if (phi_yy_     != NULL) ierr = VecDestroy(phi_yy_);                CHKERRXX(ierr);
#ifdef P4_TO_P8
    if (phi_zz_     != NULL) ierr = VecDestroy(phi_zz_);                CHKERRXX(ierr);
#endif
  }

  if (is_mue_dd_owned){
    if (mue_xx_     != NULL) ierr = VecDestroy(mue_xx_);                CHKERRXX(ierr);
    if (mue_yy_     != NULL) ierr = VecDestroy(mue_yy_);                CHKERRXX(ierr);
#ifdef P4_TO_P8
    if (mue_zz_     != NULL) ierr = VecDestroy(mue_zz_);                CHKERRXX(ierr);
#endif
  }
}

void PoissonSolverNodeBase::preallocate_matrix()
{  
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBase_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

  PetscInt num_owned_global = global_node_offset[p4est->mpisize];
  PetscInt num_owned_local  = (PetscInt)(nodes->num_owned_indeps);

  if (A != NULL)
    ierr = MatDestroy(A); CHKERRXX(ierr);

  // set up the matrix
  ierr = MatCreate(p4est->mpicomm, &A); CHKERRXX(ierr);
  ierr = MatSetType(A, MATAIJ); CHKERRXX(ierr);
  ierr = MatSetSizes(A, num_owned_local , num_owned_local,
                     num_owned_global, num_owned_global); CHKERRXX(ierr);
  ierr = MatSetFromOptions(A); CHKERRXX(ierr);

  /* preallocate space for matrix
   * This is done by computing the exact number of neighbors that will be used
   * to discretize Poisson equation which means it will adapt to T-junction
   * whenever necessary so it should have a fairly good estimate of non-zeros
   *
   * Note that this method overpredicts the actual number of non-zeros since
   * it assumes the PDE is discretized at boundary points and also points in
   * \Omega^+ that are within diag_min distance away from interface. For a
   * simple test (circle) this resulted in memory allocation for about 15%
   * extra points. Note that this does not mean there are actually 15% more
   * nonzeros, but simply that many more bytes are allocated and thus wasted.
   * This number will be smaller if there are small cells not only near the
   * interface but also inside the domain. (Note that use of worst-case estimate
   *, i.e d_nz = o_nz = 9 in 2D, in this case resulted in about 450% extra
   * memory consumption. So, getting 15% here with a simple change is a good
   * compromise here!)
   *
   * If this is still too much memory consumption, the ultimate choice is save
   * results in intermediate arrays and only allocate as much space as needed.
   * This is left for future optimizations if necessary.
   */
  std::vector<PetscInt> d_nnz(num_owned_local, 1), o_nnz(num_owned_local, 0);
  double *phi_p;
  ierr = VecGetArray(phi_, &phi_p); CHKERRXX(ierr);

  for (p4est_locidx_t n=0; n<num_owned_local; n++)
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    /*
     * Check for neighboring nodes:
     * 1) If they exist and are local nodes, increment d_nnz[n]
     * 2) If they exist but are not local nodes, increment o_nnz[n]
     * 3) If they do not exist, simply skip
     */

    if (phi_p[n] > diag_min)
      continue;

#ifdef P4_TO_P8
    if (qnnn.d_m00_p0*qnnn.d_m00_0p != 0) // node_m00_mm will enter discretization
      qnnn.node_m00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_m00_m0*qnnn.d_m00_0p != 0) // node_m00_pm will enter discretization
      qnnn.node_m00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_m00_p0*qnnn.d_m00_0m != 0) // node_m00_mp will enter discretization
      qnnn.node_m00_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_m00_m0*qnnn.d_m00_0m != 0) // node_m00_pp will enter discretization
      qnnn.node_m00_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
    if (qnnn.d_m00_p0 != 0) // node_m00_mm will enter discretization
      qnnn.node_m00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_m00_m0 != 0) // node_m00_pm will enter discretization
      qnnn.node_m00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
    if (qnnn.d_p00_p0*qnnn.d_p00_0p != 0) // node_p00_mm will enter discretization
      qnnn.node_p00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_p00_m0*qnnn.d_p00_0p != 0) // node_p00_pm will enter discretization
      qnnn.node_p00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_p00_p0*qnnn.d_p00_0m != 0) // node_p00_mp will enter discretization
      qnnn.node_p00_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_p00_m0*qnnn.d_p00_0m != 0) // node_p00_pp will enter discretization
      qnnn.node_p00_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
    if (qnnn.d_p00_p0 != 0) // node_p0_m will enter discretization
      qnnn.node_p00_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_p00_m0 != 0) // node_p0_p will enter discretization
      qnnn.node_p00_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
    if (qnnn.d_0m0_p0*qnnn.d_0m0_0p != 0) // node_0m0_mm will enter discretization
      qnnn.node_0m0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0m0_m0*qnnn.d_0m0_0p != 0) // node_0m0_pm will enter discretization
      qnnn.node_0m0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0m0_p0*qnnn.d_0m0_0m != 0) // node_0m0_mp will enter discretization
      qnnn.node_0m0_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0m0_m0*qnnn.d_0m0_0m != 0) // node_0m0_pp will enter discretization
      qnnn.node_0m0_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
    if (qnnn.d_0m0_p0 != 0) // node_0m_m will enter discretization
      qnnn.node_0m0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0m0_m0 != 0) // node_0m_p will enter discretization
      qnnn.node_0m0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
    if (qnnn.d_0p0_p0*qnnn.d_0p0_0p != 0) // node_0p0_mm will enter discretization
      qnnn.node_0p0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0p0_m0*qnnn.d_0p0_0p != 0) // node_0p0_pm will enter discretization
      qnnn.node_0p0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0p0_p0*qnnn.d_0p0_0m != 0) // node_0p0_mp will enter discretization
      qnnn.node_0p0_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0p0_m0*qnnn.d_0p0_0m != 0) // node_0p0_pp will enter discretization
      qnnn.node_0p0_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#else
    if (qnnn.d_0p0_p0 != 0) // node_0p_m will enter discretization
      qnnn.node_0p0_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_0p0_m0 != 0) // node_0p_p will enter discretization
      qnnn.node_0p0_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

#ifdef P4_TO_P8
    if (qnnn.d_00m_p0*qnnn.d_00m_0p != 0) // node_00m_mm will enter discretization
      qnnn.node_00m_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00m_m0*qnnn.d_00m_0p != 0) // node_00m_pm will enter discretization
      qnnn.node_00m_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00m_p0*qnnn.d_00m_0m != 0) // node_00m_mp will enter discretization
      qnnn.node_00m_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00m_m0*qnnn.d_00m_0m != 0) // node_00m_pp will enter discretization
      qnnn.node_00m_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;

    if (qnnn.d_00p_p0*qnnn.d_00p_0p != 0) // node_00p_mm will enter discretization
      qnnn.node_00p_mm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00p_m0*qnnn.d_00p_0p != 0) // node_00p_pm will enter discretization
      qnnn.node_00p_pm < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00p_p0*qnnn.d_00p_0m != 0) // node_00p_mp will enter discretization
      qnnn.node_00p_mp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
    if (qnnn.d_00p_m0*qnnn.d_00p_0m != 0) // node_00p_pp will enter discretization
      qnnn.node_00p_pp < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
#endif

  }

  ierr = VecRestoreArray(phi_, &phi_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBase_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void PoissonSolverNodeBase::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBase_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);

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
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  if (!is_matrix_computed)
  {
    matrix_has_nullspace = true;
    if(mue_ == NULL)
      setup_negative_laplace_matrix();
    else
      setup_negative_variable_coeff_laplace_matrix();

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
   * Plerase refer to HYPRE manual for more information on the actual importance or check Mohammad Mirzade's
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

//    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
//    if (matrix_has_nullspace){
//      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
//    }
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // setup rhs
  if (mue_ == NULL)
    setup_negative_laplace_rhsvec();
  else
    setup_negative_variable_coeff_laplace_rhsvec();

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

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBase_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);
}

void PoissonSolverNodeBase::setup_negative_laplace_matrix()
{
  preallocate_matrix();

  // register for logging purpose
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBase_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

  double eps = 1E-6*d_min*d_min;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double *v2q = p4est->connectivity->vertices;

  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p, *robin_coef_p = NULL;
  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);
  if (robin_coef_) {
    ierr = VecGetArray(robin_coef_, &robin_coef_p); CHKERRXX(ierr);
  }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {    
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    p4est_topidx_t tree_it = ni->p.piggy3.which_tree;

    double tree_xmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 0];
    double tree_ymin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 2];
#endif

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------

    double x_C  = node_x_fr_i(ni) + tree_xmin;
    double y_C  = node_y_fr_j(ni) + tree_ymin;
#ifdef P4_TO_P8
    double z_C  = node_z_fr_k(ni) + tree_zmin;
#endif

    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    double d_m00 = qnnn.d_m00;double d_p00 = qnnn.d_p00;
    double d_0m0 = qnnn.d_0m0;double d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
    double d_00m = qnnn.d_00m;double d_00p = qnnn.d_00p;
#endif

    /*
     * NOTE: All nodes are in PETSc' local numbering
     */
    double d_m00_m0=qnnn.d_m00_m0; double d_m00_p0=qnnn.d_m00_p0;
    double d_p00_m0=qnnn.d_p00_m0; double d_p00_p0=qnnn.d_p00_p0;
    double d_0m0_m0=qnnn.d_0m0_m0; double d_0m0_p0=qnnn.d_0m0_p0;
    double d_0p0_m0=qnnn.d_0p0_m0; double d_0p0_p0=qnnn.d_0p0_p0;
#ifdef P4_TO_P8
    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;
#endif

    p4est_locidx_t node_m00_mm=qnnn.node_m00_mm; p4est_locidx_t node_m00_pm=qnnn.node_m00_pm;
    p4est_locidx_t node_p00_mm=qnnn.node_p00_mm; p4est_locidx_t node_p00_pm=qnnn.node_p00_pm;
    p4est_locidx_t node_0m0_mm=qnnn.node_0m0_mm; p4est_locidx_t node_0m0_pm=qnnn.node_0m0_pm;
    p4est_locidx_t node_0p0_mm=qnnn.node_0p0_mm; p4est_locidx_t node_0p0_pm=qnnn.node_0p0_pm;
#ifdef P4_TO_P8
    p4est_locidx_t node_m00_mp=qnnn.node_m00_mp; p4est_locidx_t node_m00_pp=qnnn.node_m00_pp;
    p4est_locidx_t node_p00_mp=qnnn.node_p00_mp; p4est_locidx_t node_p00_pp=qnnn.node_p00_pp;
    p4est_locidx_t node_0m0_mp=qnnn.node_0m0_mp; p4est_locidx_t node_0m0_pp=qnnn.node_0m0_pp;
    p4est_locidx_t node_0p0_mp=qnnn.node_0p0_mp; p4est_locidx_t node_0p0_pp=qnnn.node_0p0_pp;

    p4est_locidx_t node_00m_mm=qnnn.node_00m_mm; p4est_locidx_t node_00m_mp=qnnn.node_00m_mp;
    p4est_locidx_t node_00m_pm=qnnn.node_00m_pm; p4est_locidx_t node_00m_pp=qnnn.node_00m_pp;
    p4est_locidx_t node_00p_mm=qnnn.node_00p_mm; p4est_locidx_t node_00p_mp=qnnn.node_00p_mp;
    p4est_locidx_t node_00p_pm=qnnn.node_00p_pm; p4est_locidx_t node_00p_pp=qnnn.node_00p_pp;
#endif

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

    PetscInt node_000_g = petsc_gloidx[qnnn.node_000];

    if(is_node_Wall(p4est, ni))
    {
#ifdef P4_TO_P8
      if(bc_->wallType(x_C,y_C,z_C) == DIRICHLET)
#else
      if(bc_->wallType(x_C,y_C)     == DIRICHLET)
#endif
      {
        ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
        if (phi_p[n]<0.) matrix_has_nullspace = false;
        continue;
      }
#ifdef P4_TO_P8
      if(bc_->wallType(x_C,y_C,z_C) == NEUMANN)
#else
      if(bc_->wallType(x_C,y_C)     == NEUMANN)
#endif
      {
        if(phi_p[n] > diag_min)
        {
          ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
          continue;
        }

        double w_m00, w_p00, w_0m0, w_0p0;
        PetscInt node_m00_g, node_p00_g, node_0m0_g, node_0p0_g;

        if (!is_node_xmWall(p4est, ni)){
#ifdef P4_TO_P8
          p4est_locidx_t n_m00 = d_m00_0m == 0 ? ( d_m00_m0==0 ? node_m00_mm : node_m00_pm )
                                               : ( d_m00_m0==0 ? node_m00_mp : node_m00_pp );
#else
          p4est_locidx_t n_m00 = d_m00_m0 == 0 ? node_m00_mm : node_m00_pm;
#endif
          node_m00_g  = petsc_gloidx[n_m00];
          w_m00 = -1./(d_m00*d_m00);
        }

        if (!is_node_xpWall(p4est, ni)){
#ifdef P4_TO_P8
          p4est_locidx_t n_p00 = d_p00_0m == 0 ? ( d_p00_m0 == 0 ? node_p00_mm : node_p00_pm )
                                               : ( d_p00_m0 == 0 ? node_p00_mp : node_p00_pp );
#else
          p4est_locidx_t n_p00 = d_p00_m0 == 0 ? node_p00_mm : node_p00_pm;
#endif
          node_p00_g  = petsc_gloidx[n_p00];
          w_p00 = -1./(d_p00*d_p00);
        }

        if (!is_node_ymWall(p4est, ni)){
#ifdef P4_TO_P8
          p4est_locidx_t n_0m0 = d_0m0_0m == 0 ? ( d_0m0_m0 == 0 ? node_0m0_mm : node_0m0_pm )
                                               : ( d_0m0_m0 == 0 ? node_0m0_mp : node_0m0_pp );
#else
          p4est_locidx_t n_0m0 = d_0m0_m0 == 0 ? node_0m0_mm : node_0m0_pm;
#endif
          node_0m0_g  = petsc_gloidx[n_0m0];
          w_0m0 = -1./(d_0m0*d_0m0);
        }

        if (!is_node_ypWall(p4est, ni)){
#ifdef P4_TO_P8
          p4est_locidx_t n_0p0 = d_0p0_0m == 0 ? ( d_0p0_m0 == 0 ? node_0p0_mm : node_0p0_pm )
                                               : ( d_0p0_m0 == 0 ? node_0p0_mp : node_0p0_pp );
#else
          p4est_locidx_t n_0p0 = d_0p0_m0 == 0 ? node_0p0_mm:node_0p0_pm;
#endif
          node_0p0_g  = petsc_gloidx[n_0p0];
          w_0p0 = -1./(d_0p0*d_0p0);
        }

        double diag = add_p[n]-(w_m00+w_p00+w_0m0+w_0p0);
        w_m00 /= diag;
        w_p00 /= diag;
        w_0m0 /= diag;
        w_0p0 /= diag;

        ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        if (!is_node_xmWall(p4est, ni)) { ierr = MatSetValue(A, node_000_g, node_m00_g, w_m00, ADD_VALUES); CHKERRXX(ierr); }
        if (!is_node_xpWall(p4est, ni)) { ierr = MatSetValue(A, node_000_g, node_p00_g, w_p00, ADD_VALUES); CHKERRXX(ierr); }
        if (!is_node_ymWall(p4est, ni)) { ierr = MatSetValue(A, node_000_g, node_0m0_g, w_0m0, ADD_VALUES); CHKERRXX(ierr); }
        if (!is_node_ypWall(p4est, ni)) { ierr = MatSetValue(A, node_000_g, node_0p0_g, w_0p0, ADD_VALUES); CHKERRXX(ierr); }

        if(add_p[n] > 0) matrix_has_nullspace = false;
        continue;

#ifdef P4_TO_P8
        if (is_node_zpWall(p4est, ni)){
          p4est_locidx_t n_00m = d_00m_0m == 0 ? ( d_00m_m0 == 0 ? node_00m_mm : node_00m_pm )
                                               : ( d_00m_m0 == 0 ? node_00m_mp : node_00m_pp );
          PetscInt node_00m_g  = petsc_gloidx[n_00m];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min)
            ierr = MatSetValue(A, node_000_g, node_00m_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);

          continue;
        }

        if (is_node_zmWall(p4est, ni)){
          p4est_locidx_t n_00p = d_00p_0m == 0 ? ( d_00p_m0 == 0 ? node_00p_mm : node_00p_pm )
                                               : ( d_00p_m0 == 0 ? node_00p_mp : node_00p_pp );
          PetscInt node_00p_g  = petsc_gloidx[n_00p];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min)
            ierr = MatSetValue(A, node_000_g, node_00p_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);

          continue;
        }
#endif
      }
    } else {
      double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
#ifdef P4_TO_P8
      double phi_00m, phi_00p;
#endif

#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
#endif

      //---------------------------------------------------------------------
      // interface boundary
      //---------------------------------------------------------------------
      if((ABS(phi_000)<eps && bc_->interfaceType() == DIRICHLET) ){
        ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);

        matrix_has_nullspace=false;
        continue;
      }

      // TODO: This needs optimization
#ifdef P4_TO_P8
      double P_mmm = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min, z_C-0.5*dz_min);
      double P_mpm = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min, z_C-0.5*dz_min);
      double P_pmm = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min, z_C-0.5*dz_min);
      double P_ppm = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min, z_C-0.5*dz_min);
      double P_mmp = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min, z_C+0.5*dz_min);
      double P_mpp = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min, z_C+0.5*dz_min);
      double P_pmp = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min, z_C+0.5*dz_min);
      double P_ppp = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min, z_C+0.5*dz_min);
#else
      double P_mmm = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min);
      double P_mpm = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min);
      double P_pmm = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min);
      double P_ppm = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min);
#endif

#ifdef P4_TO_P8
      bool is_one_positive = (P_mmm > 0 || P_pmm > 0 || P_mpm > 0 || P_ppm > 0 ||
                              P_mmp > 0 || P_pmp > 0 || P_mpp > 0 || P_ppp > 0 );
      bool is_one_negative = (P_mmm < 0 || P_pmm < 0 || P_mpm < 0 || P_ppm < 0 ||
                              P_mmp < 0 || P_pmp < 0 || P_mpp < 0 || P_ppp < 0 );

      bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;
#else
      bool is_ngbd_crossed_neumann = ( P_mmm*P_mpm<0 || P_mpm*P_ppm<0 || P_ppm*P_pmm<0 || P_pmm*P_mmm<0 );
#endif

      // far away from the interface
      if(phi_000>0. &&  (!is_ngbd_crossed_neumann || bc_->interfaceType() == DIRICHLET )){
        ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
        continue;
      }

      // if far away from the interface or close to it but with dirichlet
      // then finite difference method
      if ( (bc_->interfaceType() == DIRICHLET && phi_000<0.) ||
           (bc_->interfaceType() == NEUMANN   && !is_ngbd_crossed_neumann ) ||
           (bc_->interfaceType() == ROBIN     && !is_ngbd_crossed_neumann ) ||
           bc_->interfaceType() == NOINTERFACE)
      {
        double phixx_C = phi_xx_p[n];
        double phiyy_C = phi_yy_p[n];
#ifdef P4_TO_P8
        double phizz_C = phi_zz_p[n];
#endif

        bool is_interface_m00 = (bc_->interfaceType() == DIRICHLET && phi_m00*phi_000 <= 0.);
        bool is_interface_p00 = (bc_->interfaceType() == DIRICHLET && phi_p00*phi_000 <= 0.);
        bool is_interface_0m0 = (bc_->interfaceType() == DIRICHLET && phi_0m0*phi_000 <= 0.);
        bool is_interface_0p0 = (bc_->interfaceType() == DIRICHLET && phi_0p0*phi_000 <= 0.);
#ifdef P4_TO_P8
        bool is_interface_00m = (bc_->interfaceType() == DIRICHLET && phi_00m*phi_000 <= 0.);
        bool is_interface_00p = (bc_->interfaceType() == DIRICHLET && phi_00p*phi_000 <= 0.);
#endif

#ifdef P4_TO_P8
        if ( is_interface_0m0 || is_interface_m00 || is_interface_p00 || is_interface_0p0  || is_interface_00m || is_interface_00p)
#else
        if ( is_interface_0m0 || is_interface_m00 || is_interface_p00 || is_interface_0p0 )
#endif
          matrix_has_nullspace = false;

        // given boundary condition at interface from quadratic interpolation
        if( is_interface_m00) {
          double phixx_m00 = qnnn.f_m00_linear(phi_xx_p);
          double theta_m00 = interface_Location_With_Second_Order_Derivative(0., d_m00, phi_000, phi_m00, phixx_C, phixx_m00);
          if (theta_m00<eps) theta_m00 = eps; if (theta_m00>d_m00) theta_m00 = d_m00;
          d_m00_m0 = d_m00_p0 = 0;
#ifdef P4_TO_P8
          d_m00_0m = d_m00_0p = 0;
#endif
          d_m00 = theta_m00;
        }
        if( is_interface_p00){
          double phixx_p00 = qnnn.f_p00_linear(phi_xx_p);
          double theta_p00 = interface_Location_With_Second_Order_Derivative(0., d_p00, phi_000, phi_p00, phixx_C, phixx_p00);
          if (theta_p00<eps) theta_p00 = eps; if (theta_p00>d_p00) theta_p00 = d_p00;
          d_p00_m0 = d_p00_p0 = 0;
#ifdef P4_TO_P8
          d_p00_0m = d_p00_0p = 0;
#endif
          d_p00 = theta_p00;
        }
        if( is_interface_0m0){
          double phiyy_0m0 = qnnn.f_0m0_linear(phi_yy_p);
          double theta_0m0 = interface_Location_With_Second_Order_Derivative(0., d_0m0, phi_000, phi_0m0, phiyy_C, phiyy_0m0);
          if (theta_0m0<eps) theta_0m0 = eps; if (theta_0m0>d_0m0) theta_0m0 = d_0m0;
          d_0m0_m0 = d_0m0_p0 = 0;
#ifdef P4_TO_P8
          d_0m0_0m = d_0m0_0p = 0;
#endif
          d_0m0 = theta_0m0;
        }
        if( is_interface_0p0){
          double phiyy_0p0 = qnnn.f_0p0_linear(phi_yy_p);
          double theta_0p0 = interface_Location_With_Second_Order_Derivative(0., d_0p0, phi_000, phi_0p0, phiyy_C, phiyy_0p0);
          if (theta_0p0<eps) theta_0p0 = eps; if (theta_0p0>d_0p0) theta_0p0 = d_0p0;
          d_0p0_m0 = d_0p0_p0 = 0;
#ifdef P4_TO_P8
          d_0p0_0m = d_0p0_0p = 0;
#endif
          d_0p0 = theta_0p0;
        }
#ifdef P4_TO_P8
        if( is_interface_00m){
          double phizz_00m = qnnn.f_00m_linear(phi_zz_p);
          double theta_00m = interface_Location_With_Second_Order_Derivative(0., d_00m, phi_000, phi_00m, phizz_C, phizz_00m);
          if (theta_00m<eps) theta_00m = eps; if (theta_00m>d_00m) theta_00m = d_00m;
          d_00m_m0 = d_00m_p0 = d_00m_0m = d_00m_0p = 0;
          d_00m = theta_00m;
        }
        if( is_interface_00p){
          double phizz_00p = qnnn.f_00p_linear(phi_zz_p);
          double theta_00p = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000, phi_00p, phizz_C, phizz_00p);
          if (theta_00p<eps) theta_00p = eps; if (theta_00p>d_00p) theta_00p = d_00p;
          d_00p_m0 = d_00p_p0 = d_00p_0m = d_00p_0p = 0;
          d_00p = theta_00p;
        }
#endif

#ifdef P4_TO_P8
        //------------------------------------
        // Dfxx =   fxx + a*fyy + b*fzz
        // Dfyy = c*fxx +   fyy + d*fzz
        // Dfzz = e*fxx + f*fyy +   fzz
        //------------------------------------
        double a = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ;
        double b = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ;

        double c = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ;
        double d = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ;

        double e = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ;
        double f = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ;

        //------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //
        // Laplace = wi*Dfxx +
        //           wj*Dfyy +
        //           wk*Dfzz
        //------------------------------------------------------------
        double det = 1.-a*c-b*e-d*f+a*d*e+b*c*f;
        double wi = (1.-c-e+c*f+e*d-d*f)/det;
        double wj = (1.-a-f+a*e+f*b-b*e)/det;
        double wk = (1.-b-d+b*c+d*a-a*c)/det;

        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        double w_m00 = -mu_ * wi * 2./d_m00/(d_m00+d_p00);
        double w_p00 = -mu_ * wi * 2./d_p00/(d_m00+d_p00);
        double w_0m0 = -mu_ * wj * 2./d_0m0/(d_0m0+d_0p0);
        double w_0p0 = -mu_ * wj * 2./d_0p0/(d_0m0+d_0p0);
        double w_00m = -mu_ * wk * 2./d_00m/(d_00m+d_00p);
        double w_00p = -mu_ * wk * 2./d_00p/(d_00m+d_00p);

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
        w_m00 /= w_000; w_p00 /= w_000;
        w_0m0 /= w_000; w_0p0 /= w_000;
        w_00m /= w_000; w_00p /= w_000;

        //---------------------------------------------------------------------
        // add coefficients in the matrix
        //---------------------------------------------------------------------
        ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        if(!is_interface_m00)
        {
          double w_m00_mm = w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          double w_m00_mp = w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          double w_m00_pm = w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          double w_m00_pp = w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);

          if (w_m00_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_mm], w_m00_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_m00_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_mp], w_m00_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_m00_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_pm], w_m00_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_m00_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_pp], w_m00_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_p00)
        {
          double w_p00_mm = w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          double w_p00_mp = w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          double w_p00_pm = w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          double w_p00_pp = w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);

          if (w_p00_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_mm], w_p00_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_p00_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_mp], w_p00_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_p00_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_pm], w_p00_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_p00_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_pp], w_p00_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_0m0)
        {
          double w_0m0_mm = w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          double w_0m0_mp = w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          double w_0m0_pm = w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          double w_0m0_pp = w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);

          if (w_0m0_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_mm], w_0m0_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0m0_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_mp], w_0m0_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0m0_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_pm], w_0m0_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0m0_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_pp], w_0m0_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_0p0)
        {
          double w_0p0_mm = w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          double w_0p0_mp = w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          double w_0p0_pm = w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          double w_0p0_pp = w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);

          if (w_0p0_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_mm], w_0p0_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0p0_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_mp], w_0p0_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0p0_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_pm], w_0p0_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0p0_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_pp], w_0p0_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_00m)
        {
          double w_00m_mm = w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          double w_00m_mp = w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          double w_00m_pm = w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          double w_00m_pp = w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);

          if (w_00m_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_mm], w_00m_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00m_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_mp], w_00m_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00m_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_pm], w_00m_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00m_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_pp], w_00m_pp, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_00p)
        {
          double w_00p_mm = w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          double w_00p_mp = w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          double w_00p_pm = w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          double w_00p_pp = w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);

          if (w_00p_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_mm], w_00p_mm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00p_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_mp], w_00p_mp, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00p_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_pm], w_00p_pm, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00p_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_pp], w_00p_pp, ADD_VALUES); CHKERRXX(ierr);}
        }
#else
        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        double w_m00 = -2./d_m00/(d_m00+d_p00);
        double w_p00 = -2./d_p00/(d_m00+d_p00);
        double w_0m0 = -2./d_0m0/(d_0m0+d_0p0);
        double w_0p0 = -2./d_0p0/(d_0m0+d_0p0);

        //---------------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //---------------------------------------------------------------------
        double weight_on_Dyy = 1.0 - d_m00_p0*d_m00_m0/d_m00/(d_m00+d_p00) - d_p00_p0*d_p00_m0/d_p00/(d_m00+d_p00);
        double weight_on_Dxx = 1.0 - d_0m0_m0*d_0m0_p0/d_0m0/(d_0m0+d_0p0) - d_0p0_m0*d_0p0_p0/d_0p0/(d_0m0+d_0p0);

        w_m00 *= weight_on_Dxx*mu_;
        w_p00 *= weight_on_Dxx*mu_;
        w_0m0 *= weight_on_Dyy*mu_;
        w_0p0 *= weight_on_Dyy*mu_;

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------

        double diag = add_p[n]-(w_m00+w_p00+w_0m0+w_0p0);
        w_m00 /= diag;
        w_p00 /= diag;
        w_0m0 /= diag;
        w_0p0 /= diag;

        //---------------------------------------------------------------------
        // addition to diagonal elements
        //---------------------------------------------------------------------
        ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        if(!is_interface_m00) {
          PetscInt node_m00_pm_g = petsc_gloidx[node_m00_pm];
          PetscInt node_m00_mm_g = petsc_gloidx[node_m00_mm];

          if (d_m00_m0 != 0) ierr = MatSetValue(A, node_000_g, node_m00_pm_g, w_m00*d_m00_m0/(d_m00_m0+d_m00_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_m00_p0 != 0) ierr = MatSetValue(A, node_000_g, node_m00_mm_g, w_m00*d_m00_p0/(d_m00_m0+d_m00_p0), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_p00) {
          PetscInt node_p00_pm_g = petsc_gloidx[node_p00_pm];
          PetscInt node_p00_mm_g = petsc_gloidx[node_p00_mm];

          if (d_p00_m0 != 0) ierr = MatSetValue(A, node_000_g, node_p00_pm_g, w_p00*d_p00_m0/(d_p00_m0+d_p00_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_p00_p0 != 0) ierr = MatSetValue(A, node_000_g, node_p00_mm_g, w_p00*d_p00_p0/(d_p00_m0+d_p00_p0), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_0m0) {
          PetscInt node_0m0_pm_g = petsc_gloidx[node_0m0_pm];
          PetscInt node_0m0_mm_g = petsc_gloidx[node_0m0_mm];

          if (d_0m0_m0 != 0) ierr = MatSetValue(A, node_000_g, node_0m0_pm_g, w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_0m0_p0 != 0) ierr = MatSetValue(A, node_000_g, node_0m0_mm_g, w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_0p0) {
          PetscInt node_0p0_pm_g = petsc_gloidx[node_0p0_pm];
          PetscInt node_0p0_mm_g = petsc_gloidx[node_0p0_mm];

          if (d_0p0_m0 != 0) ierr = MatSetValue(A, node_000_g, node_0p0_pm_g, w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_0p0_p0 != 0) ierr = MatSetValue(A, node_000_g, node_0p0_mm_g, w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0), ADD_VALUES); CHKERRXX(ierr);
        }
#endif

        if(add_p[n] > 0) matrix_has_nullspace = false;
        continue;
      }

      // if ngbd is crossed and neumman BC
      // then use finite volume method
      // only work if the mesh is uniform close to the interface

      // FIXME: the neumann BC on the interface works only if the interface doesn't touch the edge of the domain
      if (is_ngbd_crossed_neumann && (bc_->interfaceType() == NEUMANN || bc_->interfaceType() == ROBIN) )
      {
#ifdef P4_TO_P8
        Cube3 cube;
#else
        Cube2 cube;
#endif
        cube.x0 = x_C-0.5*dx_min;
        cube.x1 = x_C+0.5*dx_min;
        cube.y0 = y_C-0.5*dy_min;
        cube.y1 = y_C+0.5*dy_min;
#ifdef P4_TO_P8
        cube.z0 = z_C-0.5*dz_min;
        cube.z1 = z_C+0.5*dz_min;
#endif
#ifdef P4_TO_P8
        OctValue  phi_cube(P_mmm, P_mmp, P_mpm, P_mpp,
                           P_pmm, P_pmp, P_ppm, P_ppp);
        double volume_cut_cell = cube.volume_In_Negative_Domain(phi_cube);
        double interface_area  = cube.interface_Area_In_Cell(phi_cube);
#else
        QuadValue phi_cube(P_mmm, P_mpm, P_pmm, P_ppm);
        double volume_cut_cell = cube.area_In_Negative_Domain(phi_cube);
        double interface_area  = cube.interface_Length_In_Cell(phi_cube);
#endif

        if (volume_cut_cell>eps*eps)
        {
#ifdef P4_TO_P8
          p4est_locidx_t quad_mmm_idx, quad_ppp_idx;
          p4est_topidx_t tree_mmm_idx, tree_ppp_idx;

          node_neighbors_->find_neighbor_cell_of_node(ni, -1, -1, -1, quad_mmm_idx, tree_mmm_idx);
          node_neighbors_->find_neighbor_cell_of_node(ni,  1,  1,  1, quad_ppp_idx, tree_ppp_idx);

          PetscInt node_m00_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpp]];
          PetscInt node_0m0_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmp]];
          PetscInt node_00m_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_ppm]];

          PetscInt node_p00_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmm]];
          PetscInt node_0p0_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpm]];
          PetscInt node_00p_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mmp]];

          Cube2 c2;
          QuadValue qv;

          c2.x0    = cube.y0 ; c2.x1    = cube.y1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mmp;    qv.val10 = P_mpm;    qv.val11 = P_mpp;
          double s_m00 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.y0 ; c2.x1    = cube.y1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_pmm;    qv.val01 = P_pmp;    qv.val10 = P_ppm;    qv.val11 = P_ppp;
          double s_p00 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mmp;    qv.val10 = P_pmm;    qv.val11 = P_pmp;
          double s_0m0 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mpm;    qv.val01 = P_mpp;    qv.val10 = P_ppm;    qv.val11 = P_ppp;
          double s_0p0 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.y0 ; c2.y1    = cube.y1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mpm;    qv.val10 = P_pmm;    qv.val11 = P_ppm;
          double s_00m = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.y0 ; c2.y1    = cube.y1 ;
          qv.val00 = P_mmp;    qv.val01 = P_mpp;    qv.val10 = P_pmp;    qv.val11 = P_ppp;
          double s_00p = c2.area_In_Negative_Domain(qv);

          double w_m00 = -mu_ * s_m00/dx_min;
          double w_p00 = -mu_ * s_p00/dx_min;
          double w_0m0 = -mu_ * s_0m0/dy_min;
          double w_0p0 = -mu_ * s_0p0/dy_min;
          double w_00m = -mu_ * s_00m/dz_min;
          double w_00p = -mu_ * s_00p/dz_min;

          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
          if (bc_->interfaceType() == ROBIN){
            if (robin_coef_p[n] > 0) matrix_has_nullspace = false;
            w_000 += robin_coef_p[n]*interface_area;
          }

          //---------------------------------------------------------------------
          // diag scaling
          //---------------------------------------------------------------------
          w_m00 /= w_000; w_p00 /= w_000;
          w_0m0 /= w_000; w_0p0 /= w_000;
          w_00m /= w_000; w_00p /= w_000;

          ierr = MatSetValue(A, node_000_g, node_000_g, 1.0,   ADD_VALUES); CHKERRXX(ierr);
          if(ABS(w_m00) > EPS) {ierr = MatSetValue(A, node_000_g, node_m00_g, w_m00, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_p00) > EPS) {ierr = MatSetValue(A, node_000_g, node_p00_g, w_p00, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_0m0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0m0_g, w_0m0, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_0p0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0p0_g, w_0p0, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_00m) > EPS) {ierr = MatSetValue(A, node_000_g, node_00m_g, w_00m, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_00p) > EPS) {ierr = MatSetValue(A, node_000_g, node_00p_g, w_00p, ADD_VALUES); CHKERRXX(ierr);}
#else

          p4est_locidx_t quad_mmm_idx, quad_ppm_idx;
          p4est_topidx_t tree_mmm_idx, tree_ppm_idx;

          node_neighbors_->find_neighbor_cell_of_node(ni, -1, -1, quad_mmm_idx, tree_mmm_idx);
          node_neighbors_->find_neighbor_cell_of_node(ni,  1,  1, quad_ppm_idx, tree_ppm_idx);

          PetscInt node_m00_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm]];
          PetscInt node_0m0_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm]];
          PetscInt node_p00_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm]];
          PetscInt node_0p0_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm]];

          double fxx,fyy;
          fxx = phi_xx_p[n];
          fyy = phi_yy_p[n];

          double s_0m0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_pmm, fxx, fxx, dx_min);
          double s_0p0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mpm, P_ppm, fxx, fxx, dx_min);
          double s_m00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_mpm, fyy, fyy, dy_min);
          double s_p00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_pmm, P_ppm, fyy, fyy, dy_min);

          double w_m00 = -mu_ * s_m00/dx_min;
          double w_p00 = -mu_ * s_p00/dx_min;
          double w_0p0 = -mu_ * s_0p0/dy_min;
          double w_0m0 = -mu_ * s_0m0/dy_min;

          double w_000 = add_p[n]*volume_cut_cell-(w_m00+w_p00+w_0m0+w_0p0);
          if (bc_->interfaceType() == ROBIN){
            if (robin_coef_p[n] > 0) matrix_has_nullspace = false;
            w_000 += robin_coef_p[n]*interface_area;
          }

          w_m00 /= w_000; w_p00 /= w_000;
          w_0m0 /= w_000; w_0p0 /= w_000;

          ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
          if (ABS(w_m00) > EPS) {ierr = MatSetValue(A, node_000_g, node_m00_g, w_m00,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_p00) > EPS) {ierr = MatSetValue(A, node_000_g, node_p00_g, w_p00,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0m0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0m0_g, w_0m0,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0p0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0p0_g, w_0p0,  ADD_VALUES); CHKERRXX(ierr);}

#endif

          if(add_p[n] > 0) matrix_has_nullspace = false;

        } else {
          ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
        }
      }
    }
  }

  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

  // restore pointers
  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);
  if (robin_coef_) {
    ierr = VecRestoreArray(robin_coef_, &robin_coef_p); CHKERRXX(ierr);
  }

  // check for null space
  // FIXME: the return value should be checked for errors ...
  MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace, 1, MPI_INT, MPI_LAND, p4est->mpicomm);
  if (matrix_has_nullspace) {
    ierr = MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE); CHKERRXX(ierr);
    p4est_gloidx_t fixed_value_idx;
    MPI_Allreduce(&fixed_value_idx_g, &fixed_value_idx, 1, MPI_LONG_LONG_INT, MPI_MIN, p4est->mpicomm);
    if (fixed_value_idx_g != fixed_value_idx){ // we are not setting the fixed value
      fixed_value_idx_l = -1;
      fixed_value_idx_g = fixed_value_idx;
    } else {
      // reset the value
      ierr = MatZeroRows(A, 1, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
    }

  }

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBase_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);
}

void PoissonSolverNodeBase::setup_negative_laplace_rhsvec()
{
  // register for logging purpose
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBase_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

  double eps = 1E-6*d_min*d_min;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double *v2q = p4est->connectivity->vertices;

  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p, *rhs_p, *robin_coef_p = NULL;
  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(rhs_,    &rhs_p   ); CHKERRXX(ierr);
  if (robin_coef_) { ierr = VecGetArray(robin_coef_, &robin_coef_p); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    p4est_topidx_t tree_it = ni->p.piggy3.which_tree;

    double tree_xmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 0];
    double tree_ymin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 2];
#endif

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------

    double x_C  = node_x_fr_i(ni) + tree_xmin;
    double y_C  = node_y_fr_j(ni) + tree_ymin;
#ifdef P4_TO_P8
    double z_C  = node_z_fr_k(ni) + tree_zmin;
#endif

    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    double d_m00 = qnnn.d_m00;double d_p00 = qnnn.d_p00;
    double d_0m0 = qnnn.d_0m0;double d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
    double d_00m = qnnn.d_00m;double d_00p = qnnn.d_00p;
#endif

    /*
     * NOTE: All nodes are in PETSc' local numbering
     */
    double d_m00_m0=qnnn.d_m00_m0; double d_m00_p0=qnnn.d_m00_p0;
    double d_p00_m0=qnnn.d_p00_m0; double d_p00_p0=qnnn.d_p00_p0;
    double d_0m0_m0=qnnn.d_0m0_m0; double d_0m0_p0=qnnn.d_0m0_p0;
    double d_0p0_m0=qnnn.d_0p0_m0; double d_0p0_p0=qnnn.d_0p0_p0;
#ifdef P4_TO_P8
    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;
#endif

    if(is_node_Wall(p4est, ni))
    {
#ifdef P4_TO_P8
      if(bc_->wallType(x_C,y_C,z_C) == DIRICHLET)
      {
        rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C);
#else
      if(bc_->wallType(x_C,y_C)     == DIRICHLET)
      {
        rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C);
#endif
        continue;
      }
#ifdef P4_TO_P8
      if(bc_->wallType(x_C,y_C,z_C) == NEUMANN)
#else
      if(bc_->wallType(x_C,y_C)     == NEUMANN)
#endif
      {
        if (is_node_xpWall(p4est, ni)){
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_m00;
#else
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*d_m00;
#endif
          continue;
        }

        if (is_node_xmWall(p4est, ni)){
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_p00;
#else
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*d_p00;
#endif
          continue;
        }

        if (is_node_ypWall(p4est, ni)){
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_0m0;
#else
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*d_0m0;
#endif
          continue;
        }
        if (is_node_ymWall(p4est, ni)){
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_0p0;
#else
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*d_0p0;
#endif
          continue;
        }
#ifdef P4_TO_P8
        if (is_node_zpWall(p4est, ni)){
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_00m;
          continue;
        }

        if (is_node_zmWall(p4est, ni)){
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_00p;
          continue;
        }
#endif
      }
    } else {
      double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
#ifdef P4_TO_P8
      double phi_00m, phi_00p;
#endif

#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
#endif

      //---------------------------------------------------------------------
      // interface boundary
      //---------------------------------------------------------------------
      if((ABS(phi_000)<eps && bc_->interfaceType() == DIRICHLET) ){
#ifdef P4_TO_P8
        rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C,z_C);
#else
        rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C);
#endif
        continue;
      }

      // TODO: This needs optimization
#ifdef P4_TO_P8
      double P_mmm = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min, z_C-0.5*dz_min);
      double P_mpm = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min, z_C-0.5*dz_min);
      double P_pmm = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min, z_C-0.5*dz_min);
      double P_ppm = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min, z_C-0.5*dz_min);
      double P_mmp = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min, z_C+0.5*dz_min);
      double P_mpp = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min, z_C+0.5*dz_min);
      double P_pmp = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min, z_C+0.5*dz_min);
      double P_ppp = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min, z_C+0.5*dz_min);
#else
      double P_mmm = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min);
      double P_mpm = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min);
      double P_pmm = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min);
      double P_ppm = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min);
#endif

#ifdef P4_TO_P8
      bool is_one_positive = (P_mmm > 0 || P_pmm > 0 || P_mpm > 0 || P_ppm > 0 ||
                              P_mmp > 0 || P_pmp > 0 || P_mpp > 0 || P_ppp > 0 );
      bool is_one_negative = (P_mmm < 0 || P_pmm < 0 || P_mpm < 0 || P_ppm < 0 ||
                              P_mmp < 0 || P_pmp < 0 || P_mpp < 0 || P_ppp < 0 );

      bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;
#else
      bool is_ngbd_crossed_neumann = ( P_mmm*P_mpm<0 || P_mpm*P_ppm<0 || P_ppm*P_pmm<0 || P_pmm*P_mmm<0 );
#endif

      // far away from the interface
      if(phi_000>0. &&  (!is_ngbd_crossed_neumann || bc_->interfaceType() == DIRICHLET )){
        if(bc_->interfaceType()==DIRICHLET)
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C,z_C);
#else
          rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C);
#endif
        else
          rhs_p[n] = 0;
        continue;
      }

      // if far away from the interface or close to it but with dirichlet
      // then finite difference method
      if ( (bc_->interfaceType() == DIRICHLET && phi_000<0.) ||
           (bc_->interfaceType() == NEUMANN   && !is_ngbd_crossed_neumann ) ||
           (bc_->interfaceType() == ROBIN     && !is_ngbd_crossed_neumann ) ||
           bc_->interfaceType() == NOINTERFACE)
      {
        double phixx_C = phi_xx_p[n];
        double phiyy_C = phi_yy_p[n];
#ifdef P4_TO_P8
        double phizz_C = phi_zz_p[n];
#endif

        bool is_interface_m00 = (bc_->interfaceType() == DIRICHLET && phi_m00*phi_000 <= 0.);
        bool is_interface_p00 = (bc_->interfaceType() == DIRICHLET && phi_p00*phi_000 <= 0.);
        bool is_interface_0m0 = (bc_->interfaceType() == DIRICHLET && phi_0m0*phi_000 <= 0.);
        bool is_interface_0p0 = (bc_->interfaceType() == DIRICHLET && phi_0p0*phi_000 <= 0.);
#ifdef P4_TO_P8
        bool is_interface_00m = (bc_->interfaceType() == DIRICHLET && phi_00m*phi_000 <= 0.);
        bool is_interface_00p = (bc_->interfaceType() == DIRICHLET && phi_00p*phi_000 <= 0.);
#endif

        double val_interface_m00 = 0.;
        double val_interface_p00 = 0.;
        double val_interface_0m0 = 0.;
        double val_interface_0p0 = 0.;
#ifdef P4_TO_P8
        double val_interface_00m = 0.;
        double val_interface_00p = 0.;
#endif

        // given boundary condition at interface from quadratic interpolation
        if( is_interface_m00) {
          double phixx_m00 = qnnn.f_m00_linear(phi_xx_p);
          double theta_m00 = interface_Location_With_Second_Order_Derivative(0., d_m00, phi_000, phi_m00, phixx_C, phixx_m00);
          if (theta_m00<eps) theta_m00 = eps; if (theta_m00>d_m00) theta_m00 = d_m00;
          d_m00_m0 = d_m00_p0 = 0;
#ifdef P4_TO_P8
          d_m00_0m = d_m00_0p = 0;
#endif
          d_m00 = theta_m00;
#ifdef P4_TO_P8
          val_interface_m00 = bc_->interfaceValue(x_C - theta_m00, y_C, z_C);
#else
          val_interface_m00 = bc_->interfaceValue(x_C - theta_m00, y_C);
#endif
        }
        if( is_interface_p00){
          double phixx_p00 = qnnn.f_p00_linear(phi_xx_p);
          double theta_p00 = interface_Location_With_Second_Order_Derivative(0., d_p00, phi_000, phi_p00, phixx_C, phixx_p00);
          if (theta_p00<eps) theta_p00 = eps; if (theta_p00>d_p00) theta_p00 = d_p00;
          d_p00_m0 = d_p00_p0 = 0;
#ifdef P4_TO_P8
          d_p00_0m = d_p00_0p = 0;
#endif
          d_p00 = theta_p00;
#ifdef P4_TO_P8
          val_interface_p00 = bc_->interfaceValue(x_C + theta_p00, y_C, z_C);
#else
          val_interface_p00 = bc_->interfaceValue(x_C + theta_p00, y_C);
#endif
        }
        if( is_interface_0m0){
          double phiyy_0m0 = qnnn.f_0m0_linear(phi_yy_p);
          double theta_0m0 = interface_Location_With_Second_Order_Derivative(0., d_0m0, phi_000, phi_0m0, phiyy_C, phiyy_0m0);
          if (theta_0m0<eps) theta_0m0 = eps; if (theta_0m0>d_0m0) theta_0m0 = d_0m0;
          d_0m0_m0 = d_0m0_p0 = 0;
#ifdef P4_TO_P8
          d_0m0_0m = d_0m0_0p = 0;
#endif
          d_0m0 = theta_0m0;
#ifdef P4_TO_P8
          val_interface_0m0 = bc_->interfaceValue(x_C, y_C - theta_0m0, z_C);
#else
          val_interface_0m0 = bc_->interfaceValue(x_C, y_C - theta_0m0);
#endif
        }
        if( is_interface_0p0){
          double phiyy_0p0 = qnnn.f_0p0_linear(phi_yy_p);
          double theta_0p0 = interface_Location_With_Second_Order_Derivative(0., d_0p0, phi_000, phi_0p0, phiyy_C, phiyy_0p0);
          if (theta_0p0<eps) theta_0p0 = eps; if (theta_0p0>d_0p0) theta_0p0 = d_0p0;
          d_0p0_m0 = d_0p0_p0 = 0;
#ifdef P4_TO_P8
          d_0p0_0m = d_0p0_0p = 0;
#endif
          d_0p0 = theta_0p0;
#ifdef P4_TO_P8
          val_interface_0p0 = bc_->interfaceValue(x_C, y_C + theta_0p0, z_C);
#else
          val_interface_0p0 = bc_->interfaceValue(x_C, y_C + theta_0p0);
#endif
        }
#ifdef P4_TO_P8
        if( is_interface_00m){
          double phizz_00m = qnnn.f_00m_linear(phi_zz_p);
          double theta_00m = interface_Location_With_Second_Order_Derivative(0., d_00m, phi_000, phi_00m, phizz_C, phizz_00m);
          if (theta_00m<eps) theta_00m = eps; if (theta_00m>d_00m) theta_00m = d_00m;
          d_00m_m0 = d_00m_p0 = d_00m_0m = d_00m_0p = 0;
          d_00m = theta_00m;
          val_interface_00m = bc_->interfaceValue(x_C, y_C , z_C - theta_00m);
        }
        if( is_interface_00p){
          double phizz_00p = qnnn.f_00p_linear(phi_zz_p);
          double theta_00p = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000, phi_00p, phizz_C, phizz_00p);
          if (theta_00p<eps) theta_00p = eps; if (theta_00p>d_00p) theta_00p = d_00p;
          d_00p_m0 = d_00p_p0 = d_00p_0m = d_00p_0p = 0;
          d_00p = theta_00p;
          val_interface_00p = bc_->interfaceValue(x_C, y_C , z_C + theta_00p);
        }
#endif

#ifdef P4_TO_P8
        //------------------------------------
        // Dfxx =   fxx + a*fyy + b*fzz
        // Dfyy = c*fxx +   fyy + d*fzz
        // Dfzz = e*fxx + f*fyy +   fzz
        //------------------------------------
        double a = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ;
        double b = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ;

        double c = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ;
        double d = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ;

        double e = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ;
        double f = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ;

        //------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //
        // Laplace = wi*Dfxx +
        //           wj*Dfyy +
        //           wk*Dfzz
        //------------------------------------------------------------
        double det = 1.-a*c-b*e-d*f+a*d*e+b*c*f;
        double wi = (1.-c-e+c*f+e*d-d*f)/det;
        double wj = (1.-a-f+a*e+f*b-b*e)/det;
        double wk = (1.-b-d+b*c+d*a-a*c)/det;

        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        double w_m00 = -mu_ * wi * 2./d_m00/(d_m00+d_p00);
        double w_p00 = -mu_ * wi * 2./d_p00/(d_m00+d_p00);
        double w_0m0 = -mu_ * wj * 2./d_0m0/(d_0m0+d_0p0);
        double w_0p0 = -mu_ * wj * 2./d_0p0/(d_0m0+d_0p0);
        double w_00m = -mu_ * wk * 2./d_00m/(d_00m+d_00p);
        double w_00p = -mu_ * wk * 2./d_00p/(d_00m+d_00p);

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
        w_m00 /= w_000; w_p00 /= w_000;
        w_0m0 /= w_000; w_0p0 /= w_000;
        w_00m /= w_000; w_00p /= w_000;

        rhs_p[n] /= w_000;

        //---------------------------------------------------------------------
        // add coefficients to the right hand side
        //---------------------------------------------------------------------
        if(is_interface_m00) rhs_p[n] -= w_m00 * val_interface_m00;
        if(is_interface_p00) rhs_p[n] -= w_p00 * val_interface_p00;
        if(is_interface_0m0) rhs_p[n] -= w_0m0 * val_interface_0m0;
        if(is_interface_0p0) rhs_p[n] -= w_0p0 * val_interface_0p0;
        if(is_interface_00m) rhs_p[n] -= w_00m * val_interface_00m;
        if(is_interface_00p) rhs_p[n] -= w_00p * val_interface_00p;
#else
        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        double w_m00 = -2./d_m00/(d_m00+d_p00);
        double w_p00 = -2./d_p00/(d_m00+d_p00);
        double w_0m0 = -2./d_0m0/(d_0m0+d_0p0);
        double w_0p0 = -2./d_0p0/(d_0m0+d_0p0);

        //---------------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //---------------------------------------------------------------------
        double weight_on_Dyy = 1.0 - d_m00_p0*d_m00_m0/d_m00/(d_m00+d_p00) - d_p00_p0*d_p00_m0/d_p00/(d_m00+d_p00);
        double weight_on_Dxx = 1.0 - d_0m0_m0*d_0m0_p0/d_0m0/(d_0m0+d_0p0) - d_0p0_m0*d_0p0_p0/d_0p0/(d_0m0+d_0p0);

        w_m00 *= weight_on_Dxx*mu_;
        w_p00 *= weight_on_Dxx*mu_;
        w_0m0 *= weight_on_Dyy*mu_;
        w_0p0 *= weight_on_Dyy*mu_;

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------

        double diag = add_p[n]-(w_m00+w_p00+w_0m0+w_0p0);
        w_m00 /= diag;
        w_p00 /= diag;
        w_0m0 /= diag;
        w_0p0 /= diag;

        rhs_p[n] /= diag;

        if(is_interface_m00) rhs_p[n] -= w_m00*val_interface_m00;
        if(is_interface_p00) rhs_p[n] -= w_p00*val_interface_p00;
        if(is_interface_0m0) rhs_p[n] -= w_0m0*val_interface_0m0;
        if(is_interface_0p0) rhs_p[n] -= w_0p0*val_interface_0p0;
#endif
        continue;
      }

      // if ngbd is crossed and neumman BC
      // then use finite volume method
      // only work if the mesh is uniform close to the interface

      // FIXME: the neumann BC on the interface works only if the interface doesn't touch the edge of the domain
      if (is_ngbd_crossed_neumann && (bc_->interfaceType() == NEUMANN || bc_->interfaceType() == ROBIN) )
      {
#ifdef P4_TO_P8
        Cube3 cube;
#else
        Cube2 cube;
#endif
        cube.x0 = x_C-0.5*dx_min;
        cube.x1 = x_C+0.5*dx_min;
        cube.y0 = y_C-0.5*dy_min;
        cube.y1 = y_C+0.5*dy_min;
#ifdef P4_TO_P8
        cube.z0 = z_C-0.5*dz_min;
        cube.z1 = z_C+0.5*dz_min;
#endif
#ifdef P4_TO_P8
        OctValue  phi_cube(P_mmm, P_mmp, P_mpm, P_mpp,
                           P_pmm, P_pmp, P_ppm, P_ppp);
        double volume_cut_cell = cube.volume_In_Negative_Domain(phi_cube);
        double interface_area  = cube.interface_Area_In_Cell(phi_cube);
#else
        QuadValue phi_cube(P_mmm, P_mpm, P_pmm, P_ppm);
        double volume_cut_cell = cube.area_In_Negative_Domain(phi_cube);
        double interface_area  = cube.interface_Length_In_Cell(phi_cube);
#endif

        if (volume_cut_cell>eps*eps)
        {
#ifdef P4_TO_P8
          Cube2 c2;
          QuadValue qv;

          c2.x0    = cube.y0 ; c2.x1    = cube.y1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mmp;    qv.val10 = P_mpm;    qv.val11 = P_mpp;
          double s_m00 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.y0 ; c2.x1    = cube.y1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_pmm;    qv.val01 = P_pmp;    qv.val10 = P_ppm;    qv.val11 = P_ppp;
          double s_p00 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mmp;    qv.val10 = P_pmm;    qv.val11 = P_pmp;
          double s_0m0 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mpm;    qv.val01 = P_mpp;    qv.val10 = P_ppm;    qv.val11 = P_ppp;
          double s_0p0 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.y0 ; c2.y1    = cube.y1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mpm;    qv.val10 = P_pmm;    qv.val11 = P_ppm;
          double s_00m = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.y0 ; c2.y1    = cube.y1 ;
          qv.val00 = P_mmp;    qv.val01 = P_mpp;    qv.val10 = P_pmp;    qv.val11 = P_ppp;
          double s_00p = c2.area_In_Negative_Domain(qv);

          double w_m00 = -mu_ * s_m00/dx_min;
          double w_p00 = -mu_ * s_p00/dx_min;
          double w_0m0 = -mu_ * s_0m0/dy_min;
          double w_0p0 = -mu_ * s_0p0/dy_min;
          double w_00m = -mu_ * s_00m/dz_min;
          double w_00p = -mu_ * s_00p/dz_min;

          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
          if (bc_->interfaceType() == ROBIN){
            w_000 += robin_coef_p[n]*interface_area;
          }

          OctValue bc_value( bc_->interfaceValue(cube.x0, cube.y0, cube.z0),
                             bc_->interfaceValue(cube.x0, cube.y0, cube.z1),
                             bc_->interfaceValue(cube.x0, cube.y1, cube.z0),
                             bc_->interfaceValue(cube.x0, cube.y1, cube.z1),
                             bc_->interfaceValue(cube.x1, cube.y0, cube.z0),
                             bc_->interfaceValue(cube.x1, cube.y0, cube.z1),
                             bc_->interfaceValue(cube.x1, cube.y1, cube.z0),
                             bc_->interfaceValue(cube.x1, cube.y1, cube.z1));

          double integral_bc = cube.integrate_Over_Interface(bc_value, phi_cube);

          rhs_p[n] *= volume_cut_cell;
          if(bc_->interfaceType() == NEUMANN)
            rhs_p[n] += mu_*integral_bc;
          else
            rhs_p[n] += integral_bc;

          rhs_p[n] /= w_000;
#else
          double fxx,fyy;
          fxx = phi_xx_p[n];
          fyy = phi_yy_p[n];

          double s_m00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_mpm, fyy, fyy, dy_min);
          double s_p00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_pmm, P_ppm, fyy, fyy, dy_min);
          double s_0m0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_pmm, fxx, fxx, dx_min);
          double s_0p0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mpm, P_ppm, fxx, fxx, dx_min);

          double w_m00 = -mu_ * s_m00/dx_min;
          double w_p00 = -mu_ * s_p00/dx_min;
          double w_0p0 = -mu_ * s_0p0/dy_min;
          double w_0m0 = -mu_ * s_0m0/dy_min;

          double w_000 = add_p[n]*volume_cut_cell-(w_m00+w_p00+w_0m0+w_0p0);
          if (bc_->interfaceType() == ROBIN){
            w_000 += robin_coef_p[n]*interface_area;
          }

          QuadValue bc_value( bc_->interfaceValue(cube.x0, cube.y0),
                              bc_->interfaceValue(cube.x0, cube.y1),
                              bc_->interfaceValue(cube.x1, cube.y0),
                              bc_->interfaceValue(cube.x1, cube.y1));

          double integral_bc = cube.integrate_Over_Interface(bc_value, phi_cube);
          rhs_p[n] *= volume_cut_cell;
          if(bc_->interfaceType() == NEUMANN)
            rhs_p[n] += mu_*integral_bc;
          else
            rhs_p[n] += integral_bc;

          rhs_p[n] /= w_000;
#endif
        } else {
          rhs_p[n] = 0.;
        }
      }
    }
  }

  if (matrix_has_nullspace && fixed_value_idx_l > 0){
    rhs_p[fixed_value_idx_l] = 0;
  }

  // restore the pointers
  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_,    &phi_p   ); CHKERRXX(ierr);
  if (robin_coef_) { ierr = VecGetArray(robin_coef_, &robin_coef_p); CHKERRXX(ierr); }

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBase_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
}

void PoissonSolverNodeBase::setup_negative_variable_coeff_laplace_matrix()
{
  preallocate_matrix();

  // register for logging purpose
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBase_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

  double eps = 1E-6*d_min*d_min;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double *v2q = p4est->connectivity->vertices;

  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p, *robin_coef_p = NULL;
  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);

  double *mue_p, *mue_xx_p, *mue_yy_p;
  ierr = VecGetArray(mue_,    &mue_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);

  double *mue_zz_p;
  ierr = VecGetArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);
  if (robin_coef_) {
    ierr = VecGetArray(robin_coef_, &robin_coef_p); CHKERRXX(ierr);
  }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    p4est_topidx_t tree_it = ni->p.piggy3.which_tree;

    double tree_xmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 0];
    double tree_ymin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 2];
#endif

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------

    double x_C  = node_x_fr_i(ni) + tree_xmin;
    double y_C  = node_y_fr_j(ni) + tree_ymin;
#ifdef P4_TO_P8
    double z_C  = node_z_fr_k(ni) + tree_zmin;
#endif

    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    double d_m00 = qnnn.d_m00;double d_p00 = qnnn.d_p00;
    double d_0m0 = qnnn.d_0m0;double d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
    double d_00m = qnnn.d_00m;double d_00p = qnnn.d_00p;
#endif

    /*
     * NOTE: All nodes are in PETSc' local numbering
     */
    double d_m00_m0=qnnn.d_m00_m0; double d_m00_p0=qnnn.d_m00_p0;
    double d_p00_m0=qnnn.d_p00_m0; double d_p00_p0=qnnn.d_p00_p0;
    double d_0m0_m0=qnnn.d_0m0_m0; double d_0m0_p0=qnnn.d_0m0_p0;
    double d_0p0_m0=qnnn.d_0p0_m0; double d_0p0_p0=qnnn.d_0p0_p0;

    double w_m00_mm, w_m00_pm;
    double w_p00_mm, w_p00_pm;
    double w_0m0_mm, w_0m0_pm;
    double w_0p0_mm, w_0p0_pm;
    double w_000, w_m00, w_p00, w_0m0, w_0p0;
#ifdef P4_TO_P8
    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;

    double w_m00_mp, w_m00_pp;
    double w_p00_mp, w_p00_pp;
    double w_0m0_mp, w_0m0_pp;
    double w_0p0_mp, w_0p0_pp;

    double w_00m_mm, w_00m_pm;
    double w_00p_mm, w_00p_pm;
    double w_00m_mp, w_00m_pp;
    double w_00p_mp, w_00p_pp;
    double w_00m, w_00p;
#endif

    p4est_locidx_t node_m00_mm=qnnn.node_m00_mm; p4est_locidx_t node_m00_pm=qnnn.node_m00_pm;
    p4est_locidx_t node_p00_mm=qnnn.node_p00_mm; p4est_locidx_t node_p00_pm=qnnn.node_p00_pm;
    p4est_locidx_t node_0m0_mm=qnnn.node_0m0_mm; p4est_locidx_t node_0m0_pm=qnnn.node_0m0_pm;
    p4est_locidx_t node_0p0_mm=qnnn.node_0p0_mm; p4est_locidx_t node_0p0_pm=qnnn.node_0p0_pm;

#ifdef P4_TO_P8
    p4est_locidx_t node_m00_mp=qnnn.node_m00_mp; p4est_locidx_t node_m00_pp=qnnn.node_m00_pp;
    p4est_locidx_t node_p00_mp=qnnn.node_p00_mp; p4est_locidx_t node_p00_pp=qnnn.node_p00_pp;
    p4est_locidx_t node_0m0_mp=qnnn.node_0m0_mp; p4est_locidx_t node_0m0_pp=qnnn.node_0m0_pp;
    p4est_locidx_t node_0p0_mp=qnnn.node_0p0_mp; p4est_locidx_t node_0p0_pp=qnnn.node_0p0_pp;

    p4est_locidx_t node_00m_mm=qnnn.node_00m_mm; p4est_locidx_t node_00m_mp=qnnn.node_00m_mp;
    p4est_locidx_t node_00m_pm=qnnn.node_00m_pm; p4est_locidx_t node_00m_pp=qnnn.node_00m_pp;
    p4est_locidx_t node_00p_mm=qnnn.node_00p_mm; p4est_locidx_t node_00p_mp=qnnn.node_00p_mp;
    p4est_locidx_t node_00p_pm=qnnn.node_00p_pm; p4est_locidx_t node_00p_pp=qnnn.node_00p_pp;
#endif

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

    PetscInt node_000_g = petsc_gloidx[qnnn.node_000];

    if(is_node_Wall(p4est, ni))
    {
#ifdef P4_TO_P8
      if(bc_->wallType(x_C,y_C,z_C) == DIRICHLET)
#else
      if(bc_->wallType(x_C,y_C)     == DIRICHLET)
#endif
      {
        // FIXME: get rid of bc_strength alltogether!
        ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
        if (phi_p[n]<0.) matrix_has_nullspace = false;
        continue;
      }
      // FIXME: make the neumann on the boundary second-order accurate
#ifdef P4_TO_P8
      if(bc_->wallType(x_C,y_C,z_C) == NEUMANN)
#else
      if(bc_->wallType(x_C,y_C)     == NEUMANN)
#endif
      {
        if (is_node_xpWall(p4est, ni)){
#ifdef P4_TO_P8
          p4est_locidx_t n_m00 = d_m00_0m == 0 ? ( d_m00_m0==0 ? node_m00_mm : node_m00_pm )
                                               : ( d_m00_m0==0 ? node_m00_mp : node_m00_pp );
#else
          p4est_locidx_t n_m00 = d_m00_m0 == 0 ? node_m00_mm : node_m00_pm;
#endif
          PetscInt node_m00_g  = petsc_gloidx[n_m00];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);

          if (phi_p[n] < diag_min)
            ierr = MatSetValue(A, node_000_g, node_m00_g, -bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);

          continue;
        }

        if (is_node_xmWall(p4est, ni)){
#ifdef P4_TO_P8
          p4est_locidx_t n_p00 = d_p00_0m == 0 ? ( d_p00_m0 == 0 ? node_p00_mm : node_p00_pm )
                                               : ( d_p00_m0 == 0 ? node_p00_mp : node_p00_pp );
#else
          p4est_locidx_t n_p00 = d_p00_m0 == 0 ? node_p00_mm : node_p00_pm;
#endif
          PetscInt node_p00_g  = petsc_gloidx[n_p00];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min)
            ierr = MatSetValue(A, node_000_g, node_p00_g, -bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);

          continue;
        }

        if (is_node_ypWall(p4est, ni)){
#ifdef P4_TO_P8
          p4est_locidx_t n_0m0 = d_0m0_0m == 0 ? ( d_0m0_m0 == 0 ? node_0m0_mm : node_0m0_pm )
                                               : ( d_0m0_m0 == 0 ? node_0m0_mp : node_0m0_pp );
#else
          p4est_locidx_t n_0m0 = d_0m0_m0 == 0 ? node_0m0_mm : node_0m0_pm;
#endif
          PetscInt node_0m0_g  = petsc_gloidx[n_0m0];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min)
            ierr = MatSetValue(A, node_000_g, node_0m0_g, -bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);

          continue;
        }
        if (is_node_ymWall(p4est, ni)){
#ifdef P4_TO_P8
          p4est_locidx_t n_0p0 = d_0p0_0m == 0 ? ( d_0p0_m0 == 0 ? node_0p0_mm : node_0p0_pm )
                                               : ( d_0p0_m0 == 0 ? node_0p0_mp : node_0p0_pp );
#else
          p4est_locidx_t n_0p0 = d_0p0_m0 == 0 ? node_0p0_mm:node_0p0_pm;
#endif
          PetscInt node_0p0_g  = petsc_gloidx[n_0p0];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min)
            ierr = MatSetValue(A, node_000_g, node_0p0_g, -bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);

          continue;
        }
#ifdef P4_TO_P8
        if (is_node_zpWall(p4est, ni)){
          p4est_locidx_t n_00m = d_00m_0m == 0 ? ( d_00m_m0 == 0 ? node_00m_mm : node_00m_pm )
                                               : ( d_00m_m0 == 0 ? node_00m_mp : node_00m_pp );
          PetscInt node_00m_g  = petsc_gloidx[n_00m];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min)
            ierr = MatSetValue(A, node_000_g, node_00m_g, -bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);

          continue;
        }

        if (is_node_zmWall(p4est, ni)){
          p4est_locidx_t n_00p = d_00p_0m == 0 ? ( d_00p_m0 == 0 ? node_00p_mm : node_00p_pm )
                                               : ( d_00p_m0 == 0 ? node_00p_mp : node_00p_pp );
          PetscInt node_00p_g  = petsc_gloidx[n_00p];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min)
            ierr = MatSetValue(A, node_000_g, node_00p_g, -bc_strength*mue_p[n], ADD_VALUES); CHKERRXX(ierr);

          continue;
        }
#endif
      }
    } else {
      double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
      double mue_000, mue_p00, mue_m00, mue_0m0, mue_0p0;

#ifdef P4_TO_P8
      double phi_00m, phi_00p;
      double mue_00m, mue_00p;
#endif

#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
      qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
      qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0);
#endif

      //---------------------------------------------------------------------
      // interface boundary
      //---------------------------------------------------------------------
      if((ABS(phi_000)<eps && bc_->interfaceType() == DIRICHLET) ){
        ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);

        matrix_has_nullspace=false;
        continue;
      }

      // TODO: This needs optimization
#ifdef P4_TO_P8
      double P_mmm = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min, z_C-0.5*dz_min);
      double P_mpm = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min, z_C-0.5*dz_min);
      double P_pmm = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min, z_C-0.5*dz_min);
      double P_ppm = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min, z_C-0.5*dz_min);
      double P_mmp = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min, z_C+0.5*dz_min);
      double P_mpp = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min, z_C+0.5*dz_min);
      double P_pmp = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min, z_C+0.5*dz_min);
      double P_ppp = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min, z_C+0.5*dz_min);
#else
      double P_mmm = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min);
      double P_mpm = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min);
      double P_pmm = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min);
      double P_ppm = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min);
#endif

#ifdef P4_TO_P8
      bool is_one_positive = (P_mmm > 0 || P_pmm > 0 || P_mpm > 0 || P_ppm > 0 ||
                              P_mmp > 0 || P_pmp > 0 || P_mpp > 0 || P_ppp > 0 );
      bool is_one_negative = (P_mmm < 0 || P_pmm < 0 || P_mpm < 0 || P_ppm < 0 ||
                              P_mmp < 0 || P_pmp < 0 || P_mpp < 0 || P_ppp < 0 );

      bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;
#else
      bool is_ngbd_crossed_neumann = ( P_mmm*P_mpm<0 || P_mpm*P_ppm<0 || P_ppm*P_pmm<0 || P_pmm*P_mmm<0 );
#endif

      // far away from the interface
      if(phi_000>0. &&  (!is_ngbd_crossed_neumann || bc_->interfaceType() == DIRICHLET )){
        ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
        continue;
      }

      // if far away from the interface or close to it but with dirichlet
      // then finite difference method
      if ( (bc_->interfaceType() == DIRICHLET && phi_000<0.) ||
           (bc_->interfaceType() == NEUMANN   && !is_ngbd_crossed_neumann ) ||
           (bc_->interfaceType() == ROBIN     && !is_ngbd_crossed_neumann ) ||
            bc_->interfaceType() == NOINTERFACE)
      {
        double phixx_C = phi_xx_p[n];
        double phiyy_C = phi_yy_p[n];
#ifdef P4_TO_P8
        double phizz_C = phi_zz_p[n];
#endif

        bool is_interface_m00 = (bc_->interfaceType() == DIRICHLET && phi_m00*phi_000 <= 0.);
        bool is_interface_p00 = (bc_->interfaceType() == DIRICHLET && phi_p00*phi_000 <= 0.);
        bool is_interface_0m0 = (bc_->interfaceType() == DIRICHLET && phi_0m0*phi_000 <= 0.);
        bool is_interface_0p0 = (bc_->interfaceType() == DIRICHLET && phi_0p0*phi_000 <= 0.);
#ifdef P4_TO_P8
        bool is_interface_00m = (bc_->interfaceType() == DIRICHLET && phi_00m*phi_000 <= 0.);
        bool is_interface_00p = (bc_->interfaceType() == DIRICHLET && phi_00p*phi_000 <= 0.);
#endif

#ifdef P4_TO_P8
        if ( is_interface_0m0 || is_interface_m00 || is_interface_p00 || is_interface_0p0  || is_interface_00m || is_interface_00p)
#else
        if ( is_interface_0m0 || is_interface_m00 || is_interface_p00 || is_interface_0p0 )
#endif
          matrix_has_nullspace = false;

        // given boundary condition at interface from quadratic interpolation
        if( is_interface_m00) {
          double phixx_m00 = qnnn.f_m00_linear(phi_xx_p);
          double theta_m00 = interface_Location_With_Second_Order_Derivative(0., d_m00, phi_000, phi_m00, phixx_C, phixx_m00);
          if (theta_m00<eps) theta_m00 = eps; if (theta_m00>d_m00) theta_m00 = d_m00;
          d_m00_m0 = d_m00_p0 = 0;
#ifdef P4_TO_P8
          d_m00_0m = d_m00_0p = 0;
#endif

          double mxx_000 = mue_xx_p[n];
          double mxx_m00 = qnnn.f_m00_linear(mue_xx_p);
          mue_m00 = mue_000*(1-theta_m00/d_m00) + mue_m00*theta_m00/d_m00 + 0.5*theta_m00*(theta_m00-d_m00)*MINMOD(mxx_m00,mxx_000);

          d_m00 = theta_m00;
        }
        if( is_interface_p00){
          double phixx_p00 = qnnn.f_p00_linear(phi_xx_p);
          double theta_p00 = interface_Location_With_Second_Order_Derivative(0., d_p00, phi_000, phi_p00, phixx_C, phixx_p00);
          if (theta_p00<eps) theta_p00 = eps; if (theta_p00>d_p00) theta_p00 = d_p00;
          d_p00_m0 = d_p00_p0 = 0;
#ifdef P4_TO_P8
          d_p00_0m = d_p00_0p = 0;
#endif

          double mxx_000 = mue_xx_p[n];
          double mxx_p00 = qnnn.f_p00_linear(mue_xx_p);
          mue_p00 = mue_000*(1-theta_p00/d_p00) + mue_p00*theta_p00/d_p00 + 0.5*theta_p00*(theta_p00-d_p00)*MINMOD(mxx_p00,mxx_000);

          d_p00 = theta_p00;
        }
        if( is_interface_0m0){
          double phiyy_0m0 = qnnn.f_0m0_linear(phi_yy_p);
          double theta_0m0 = interface_Location_With_Second_Order_Derivative(0., d_0m0, phi_000, phi_0m0, phiyy_C, phiyy_0m0);
          if (theta_0m0<eps) theta_0m0 = eps; if (theta_0m0>d_0m0) theta_0m0 = d_0m0;
          d_0m0_m0 = d_0m0_p0 = 0;
#ifdef P4_TO_P8
          d_0m0_0m = d_0m0_0p = 0;
#endif

          double myy_000 = mue_yy_p[n];
          double myy_0m0 = qnnn.f_0m0_linear(mue_yy_p);
          mue_0m0 = mue_000*(1-theta_0m0/d_0m0) + mue_0m0*theta_0m0/d_0m0 + 0.5*theta_0m0*(theta_0m0-d_0m0)*MINMOD(myy_0m0,myy_000);

          d_0m0 = theta_0m0;
        }
        if( is_interface_0p0){
          double phiyy_0p0 = qnnn.f_0p0_linear(phi_yy_p);
          double theta_0p0 = interface_Location_With_Second_Order_Derivative(0., d_0p0, phi_000, phi_0p0, phiyy_C, phiyy_0p0);
          if (theta_0p0<eps) theta_0p0 = eps; if (theta_0p0>d_0p0) theta_0p0 = d_0p0;
          d_0p0_m0 = d_0p0_p0 = 0;
#ifdef P4_TO_P8
          d_0p0_0m = d_0p0_0p = 0;
#endif

          double myy_000 = mue_yy_p[n];
          double myy_0p0 = qnnn.f_0p0_linear(mue_yy_p);
          mue_0p0 = mue_000*(1-theta_0p0/d_0p0) + mue_0p0*theta_0p0/d_0p0 + 0.5*theta_0p0*(theta_0p0-d_0p0)*MINMOD(myy_0p0,myy_000);

          d_0p0 = theta_0p0;
        }
#ifdef P4_TO_P8
        if( is_interface_00m){
          double phizz_00m = qnnn.f_00m_linear(phi_zz_p);
          double theta_00m = interface_Location_With_Second_Order_Derivative(0., d_00m, phi_000, phi_00m, phizz_C, phizz_00m);
          if (theta_00m<eps) theta_00m = eps; if (theta_00m>d_00m) theta_00m = d_00m;
          d_00m_m0 = d_00m_p0 = d_00m_0m = d_00m_0p = 0;

          double mzz_000 = mue_zz_p[n];
          double mzz_00m = qnnn.f_00m_linear(mue_zz_p);
          mue_00m = mue_000*(1-theta_00m/d_00m) + mue_00m*theta_00m/d_00m + 0.5*theta_00m*(theta_00m-d_00m)*MINMOD(mzz_00m,mzz_000);

          d_00m = theta_00m;
        }
        if( is_interface_00p){
          double phizz_00p = qnnn.f_00p_linear(phi_zz_p);
          double theta_00p = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000, phi_00p, phizz_C, phizz_00p);
          if (theta_00p<eps) theta_00p = eps; if (theta_00p>d_00p) theta_00p = d_00p;
          d_00p_m0 = d_00p_p0 = d_00p_0m = d_00p_0p = 0;

          double mzz_000 = mue_zz_p[n];
          double mzz_00p = qnnn.f_00p_linear(mue_zz_p);
          mue_00p = mue_000*(1-theta_00p/d_00p) + mue_00p*theta_00p/d_00p + 0.5*theta_00p*(theta_00p-d_00p)*MINMOD(mzz_00p,mzz_000);

          d_00p = theta_00p;
        }
#endif

#ifdef P4_TO_P8
        //------------------------------------
        // Dfxx =   fxx + a*fyy + b*fzz
        // Dfyy = c*fxx +   fyy + d*fzz
        // Dfzz = e*fxx + f*fyy +   fzz
        //------------------------------------
        double a = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ;
        double b = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ;

        double c = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ;
        double d = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ;

        double e = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ;
        double f = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ;

        //------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //
        // Laplace = wi*Dfxx +
        //           wj*Dfyy +
        //           wk*Dfzz
        //------------------------------------------------------------
        double det = 1.-a*c-b*e-d*f+a*d*e+b*c*f;
        double wi = (1.-c-e+c*f+e*d-d*f)/det;
        double wj = (1.-a-f+a*e+f*b-b*e)/det;
        double wk = (1.-b-d+b*c+d*a-a*c)/det;

        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        w_m00 = -2.0*wi/d_m00/(d_m00+d_p00);
        w_p00 = -2.0*wi/d_p00/(d_m00+d_p00);
        w_0m0 = -2.0*wj/d_0m0/(d_0m0+d_0p0);
        w_0p0 = -2.0*wj/d_0p0/(d_0m0+d_0p0);
        w_00m = -2.0*wk/d_00m/(d_00m+d_00p);
        w_00p = -2.0*wk/d_00p/(d_00m+d_00p);

        if(!is_interface_m00) {
          w_m00_mm = 0.5*(mue_000 + mue_p[node_m00_mm])*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00_mp = 0.5*(mue_000 + mue_p[node_m00_mp])*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00_pm = 0.5*(mue_000 + mue_p[node_m00_pm])*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00_pp = 0.5*(mue_000 + mue_p[node_m00_pp])*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
          w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
        } else {
          w_m00 *= 0.5*(mue_000 + mue_m00);
        }

        if(!is_interface_p00) {
          w_p00_mm = 0.5*(mue_000 + mue_p[node_p00_mm])*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00_mp = 0.5*(mue_000 + mue_p[node_p00_mp])*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00_pm = 0.5*(mue_000 + mue_p[node_p00_pm])*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00_pp = 0.5*(mue_000 + mue_p[node_p00_pp])*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
          w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
        } else {
          w_p00 *= 0.5*(mue_000 + mue_p00);
        }

        if(!is_interface_0m0) {
          w_0m0_mm = 0.5*(mue_000 + mue_p[node_0m0_mm])*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0_mp = 0.5*(mue_000 + mue_p[node_0m0_mp])*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0_pm = 0.5*(mue_000 + mue_p[node_0m0_pm])*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0_pp = 0.5*(mue_000 + mue_p[node_0m0_pp])*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
          w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
        } else {
          w_0m0 *= 0.5*(mue_000 + mue_0m0);
        }

        if(!is_interface_0p0) {
          w_0p0_mm = 0.5*(mue_000 + mue_p[node_0p0_mm])*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0_mp = 0.5*(mue_000 + mue_p[node_0p0_mp])*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0_pm = 0.5*(mue_000 + mue_p[node_0p0_pm])*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0_pp = 0.5*(mue_000 + mue_p[node_0p0_pp])*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
          w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
        } else {
          w_0p0 *= 0.5*(mue_000 + mue_0p0);
        }

        if(!is_interface_00m) {
          w_00m_mm = 0.5*(mue_000 + mue_p[node_00m_mm])*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m_mp = 0.5*(mue_000 + mue_p[node_00m_mp])*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m_pm = 0.5*(mue_000 + mue_p[node_00m_pm])*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m_pp = 0.5*(mue_000 + mue_p[node_00m_pp])*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
          w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
        } else {
          w_00m *= 0.5*(mue_000 + mue_00m);
        }

        if(!is_interface_00p) {
          w_00p_mm = 0.5*(mue_000 + mue_p[node_00p_mm])*w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p_mp = 0.5*(mue_000 + mue_p[node_00p_mp])*w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p_pm = 0.5*(mue_000 + mue_p[node_00p_pm])*w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p_pp = 0.5*(mue_000 + mue_p[node_00p_pp])*w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
          w_00p = w_00p_mm + w_00p_mp + w_00p_pm + w_00p_pp;
        } else {
          w_00p *= 0.5*(mue_000 + mue_00p);
        }
#else
        //---------------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //---------------------------------------------------------------------
        double wi = 1.0 - d_0m0_m0*d_0m0_p0/d_0m0/(d_0m0+d_0p0) - d_0p0_m0*d_0p0_p0/d_0p0/(d_0m0+d_0p0);
        double wj = 1.0 - d_m00_p0*d_m00_m0/d_m00/(d_m00+d_p00) - d_p00_p0*d_p00_m0/d_p00/(d_m00+d_p00);

        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        w_m00 = -2.0*wi/d_m00/(d_m00+d_p00);
        w_p00 = -2.0*wi/d_p00/(d_m00+d_p00);
        w_0m0 = -2.0*wj/d_0m0/(d_0m0+d_0p0);
        w_0p0 = -2.0*wj/d_0p0/(d_0m0+d_0p0);

        //---------------------------------------------------------------------
        // addition to diagonal elements
        //---------------------------------------------------------------------
        if(!is_interface_m00) {
          w_m00_mm = 0.5*(mue_000 + mue_p[node_m00_mm])*w_m00*d_m00_p0/(d_m00_m0+d_m00_p0);
          w_m00_pm = 0.5*(mue_000 + mue_p[node_m00_pm])*w_m00*d_m00_m0/(d_m00_m0+d_m00_p0);
          w_m00 = w_m00_mm + w_m00_pm;
        } else {
          w_m00 *= 0.5*(mue_000 + mue_m00);
        }

        if(!is_interface_p00) {
          w_p00_mm = 0.5*(mue_000 + mue_p[node_p00_mm])*w_p00*d_p00_p0/(d_p00_m0+d_p00_p0);
          w_p00_pm = 0.5*(mue_000 + mue_p[node_p00_pm])*w_p00*d_p00_m0/(d_p00_m0+d_p00_p0);
          w_p00    = w_p00_mm + w_p00_pm;
        } else {
          w_p00 *= 0.5*(mue_000 + mue_p00);
        }

        if(!is_interface_0m0) {
          w_0m0_mm = 0.5*(mue_000 + mue_p[node_0m0_mm])*w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0);
          w_0m0_pm = 0.5*(mue_000 + mue_p[node_0m0_pm])*w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0);
          w_0m0 = w_0m0_mm + w_0m0_pm;
        } else {
          w_0m0 *= 0.5*(mue_000 + mue_0m0);
        }

        if(!is_interface_0p0) {
          w_0p0_mm = 0.5*(mue_000 + mue_p[node_0p0_mm])*w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0);
          w_0p0_pm = 0.5*(mue_000 + mue_p[node_0p0_pm])*w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0);
          w_0p0 = w_0p0_mm + w_0p0_pm;
        } else {
          w_0p0 *= 0.5*(mue_000 + mue_0p0);
        }
#endif

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------

#ifdef P4_TO_P8
        w_000  = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
#else
        w_000  = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 );
#endif
        w_m00 /= w_000; w_p00 /= w_000;
        w_0m0 /= w_000; w_0p0 /= w_000;
#ifdef P4_TO_P8
        w_00m /= w_000; w_00p /= w_000;
#endif

        //---------------------------------------------------------------------
        // add coefficients in the matrix
        //---------------------------------------------------------------------
        if (node_000_g < fixed_value_idx_g){
          fixed_value_idx_l = n;
          fixed_value_idx_g = node_000_g;
        }
        ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        if(!is_interface_m00) {
          // FIXME: chaneg the direct comparison to a tolerance
          if (w_m00_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_mm], w_m00_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_m00_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_pm], w_m00_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (w_m00_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_mp], w_m00_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_m00_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_pp], w_m00_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }

        if(!is_interface_p00) {
          if (w_p00_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_mm], w_p00_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_p00_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_pm], w_p00_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (w_p00_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_mp], w_p00_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_p00_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_pp], w_p00_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }

        if(!is_interface_0m0) {
          if (w_0m0_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_mm], w_0m0_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0m0_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_pm], w_0m0_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (w_0m0_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_mp], w_0m0_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0m0_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_pp], w_0m0_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }

        if(!is_interface_0p0) {
          if (w_0p0_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_mm], w_0p0_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0p0_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_pm], w_0p0_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (w_0p0_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_mp], w_0p0_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_0p0_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_pp], w_0p0_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }
#ifdef P4_TO_P8
        if(!is_interface_00m) {
          if (w_00m_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_mm], w_00m_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00m_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_pm], w_00m_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00m_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_mp], w_00m_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00m_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_pp], w_00m_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_00p) {
          if (w_00p_mm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_mm], w_00p_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00p_pm != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_pm], w_00p_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00p_mp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_mp], w_00p_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (w_00p_pp != 0) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_pp], w_00p_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
        }
#endif

        if(add_p[n] > 0) matrix_has_nullspace = false;
        continue;
      }

      // if ngbd is crossed and neumman BC
      // then use finite volume method
      // only work if the mesh is uniform close to the interface

      // FIXME: the neumann BC on the interface works only if the interface doesn't touch the edge of the domain
      if (is_ngbd_crossed_neumann && (bc_->interfaceType() == NEUMANN || bc_->interfaceType() == ROBIN))
      {
#ifdef P4_TO_P8
        Cube3 cube;
#else
        Cube2 cube;
#endif
        cube.x0 = x_C-0.5*dx_min;
        cube.x1 = x_C+0.5*dx_min;
        cube.y0 = y_C-0.5*dy_min;
        cube.y1 = y_C+0.5*dy_min;
#ifdef P4_TO_P8
        cube.z0 = z_C-0.5*dz_min;
        cube.z1 = z_C+0.5*dz_min;
#endif
#ifdef P4_TO_P8
        OctValue  phi_cube(P_mmm, P_mmp, P_mpm, P_mpp,
                           P_pmm, P_pmp, P_ppm, P_ppp);
        double volume_cut_cell = cube.volume_In_Negative_Domain(phi_cube);
        double interface_area  = cube.interface_Area_In_Cell(phi_cube);
#else
        QuadValue phi_cube(P_mmm, P_mpm, P_pmm, P_ppm);
        double volume_cut_cell = cube.area_In_Negative_Domain(phi_cube);
        double interface_area  = cube.interface_Length_In_Cell(phi_cube);
#endif

        if (volume_cut_cell>eps*eps)
        {
#ifdef P4_TO_P8
          p4est_locidx_t quad_mmm_idx, quad_ppp_idx;
          p4est_topidx_t tree_mmm_idx, tree_ppp_idx;

          node_neighbors_->find_neighbor_cell_of_node(ni, -1, -1, -1, quad_mmm_idx, tree_mmm_idx);
          node_neighbors_->find_neighbor_cell_of_node(ni,  1,  1,  1, quad_ppp_idx, tree_ppp_idx);

          PetscInt node_m00_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpp]];
          PetscInt node_0m0_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmp]];
          PetscInt node_00m_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_ppm]];

          PetscInt node_p00_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_pmm]];
          PetscInt node_0p0_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mpm]];
          PetscInt node_00p_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppp_idx + dir::v_mmp]];

          Cube2 c2;
          QuadValue qv;

          c2.x0    = cube.y0 ; c2.x1    = cube.y1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mmp;    qv.val10 = P_mpm;    qv.val11 = P_mpp;
          double s_m00 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.y0 ; c2.x1    = cube.y1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_pmm;    qv.val01 = P_pmp;    qv.val10 = P_ppm;    qv.val11 = P_ppp;
          double s_p00 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mmp;    qv.val10 = P_pmm;    qv.val11 = P_pmp;
          double s_0m0 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mpm;    qv.val01 = P_mpp;    qv.val10 = P_ppm;    qv.val11 = P_ppp;
          double s_0p0 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.y0 ; c2.y1    = cube.y1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mpm;    qv.val10 = P_pmm;    qv.val11 = P_ppm;
          double s_00m = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.y0 ; c2.y1    = cube.y1 ;
          qv.val00 = P_mmp;    qv.val01 = P_mpp;    qv.val10 = P_pmp;    qv.val11 = P_ppp;
          double s_00p = c2.area_In_Negative_Domain(qv);

          double w_m00 = -mue_m00 * s_m00/dx_min;
          double w_p00 = -mue_p00 * s_p00/dx_min;
          double w_0m0 = -mue_0m0 * s_0m0/dy_min;
          double w_0p0 = -mue_0p0 * s_0p0/dy_min;
          double w_00m = -mue_00m * s_00m/dz_min;
          double w_00p = -mue_00p * s_00p/dz_min;

          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
          if (bc_->interfaceType() == ROBIN){
            if (robin_coef_p[n] > 0) matrix_has_nullspace = false;
            w_000 += robin_coef_p[n]*interface_area;
          }

          //---------------------------------------------------------------------
          // diag scaling
          //---------------------------------------------------------------------
          w_m00 /= w_000; w_p00 /= w_000;
          w_0m0 /= w_000; w_0p0 /= w_000;
          w_00m /= w_000; w_00p /= w_000;

          ierr = MatSetValue(A, node_000_g, node_000_g, 1.0,   ADD_VALUES); CHKERRXX(ierr);
          if(ABS(w_m00) > EPS) {ierr = MatSetValue(A, node_000_g, node_m00_g, w_m00, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_p00) > EPS) {ierr = MatSetValue(A, node_000_g, node_p00_g, w_p00, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_0m0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0m0_g, w_0m0, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_0p0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0p0_g, w_0p0, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_00m) > EPS) {ierr = MatSetValue(A, node_000_g, node_00m_g, w_00m, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_00p) > EPS) {ierr = MatSetValue(A, node_000_g, node_00p_g, w_00p, ADD_VALUES); CHKERRXX(ierr);}
#else
          p4est_locidx_t quad_mmm_idx, quad_ppm_idx;
          p4est_topidx_t tree_mmm_idx, tree_ppm_idx;

          node_neighbors_->find_neighbor_cell_of_node(ni, -1, -1, quad_mmm_idx, tree_mmm_idx);
          node_neighbors_->find_neighbor_cell_of_node(ni,  1,  1, quad_ppm_idx, tree_ppm_idx);

          PetscInt node_m00_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_mpm]];
          PetscInt node_0m0_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_mmm_idx + dir::v_pmm]];
          PetscInt node_p00_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_pmm]];
          PetscInt node_0p0_g = petsc_gloidx[nodes->local_nodes[P4EST_CHILDREN*quad_ppm_idx + dir::v_mpm]];

          double fxx,fyy;
          fxx = phi_xx_p[n];
          fyy = phi_yy_p[n];

          double s_0m0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_pmm, fxx, fxx, dx_min);
          double s_0p0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mpm, P_ppm, fxx, fxx, dx_min);
          double s_m00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_mpm, fyy, fyy, dy_min);
          double s_p00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_pmm, P_ppm, fyy, fyy, dy_min);

          double w_m00 = -mue_m00 * s_m00/dx_min;
          double w_p00 = -mue_p00 * s_p00/dx_min;
          double w_0p0 = -mue_0m0 * s_0p0/dy_min;
          double w_0m0 = -mue_0p0 * s_0m0/dy_min;

          double w_000 = add_p[n]*volume_cut_cell-(w_m00+w_p00+w_0m0+w_0p0);          
          if (bc_->interfaceType() == ROBIN){
            if (robin_coef_p[n] > 0) matrix_has_nullspace = false;
            w_000 += robin_coef_p[n]*interface_area;
          }

          w_m00 /= w_000; w_p00 /= w_000;
          w_0m0 /= w_000; w_0p0 /= w_000;

          ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
          if (ABS(w_m00) > EPS) {ierr = MatSetValue(A, node_000_g, node_m00_g, w_m00,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_p00) > EPS) {ierr = MatSetValue(A, node_000_g, node_p00_g, w_p00,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0m0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0m0_g, w_0m0,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0p0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0p0_g, w_0p0,  ADD_VALUES); CHKERRXX(ierr);}

#endif

          if(add_p[n] > 0) matrix_has_nullspace = false;

        } else {
          ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);          
        }
      }
    }
  }

  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

  // restore pointers
  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(mue_,    &mue_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);
  if (robin_coef_) {
    ierr = VecRestoreArray(robin_coef_, &robin_coef_p); CHKERRXX(ierr);
  }

  // check for null space
  // FIXME: the return value should be checked for errors ...
  MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace, 1, MPI_INT, MPI_LAND, p4est->mpicomm);
  if (matrix_has_nullspace) {
    ierr = MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE); CHKERRXX(ierr);
    p4est_gloidx_t fixed_value_idx;
    MPI_Allreduce(&fixed_value_idx_g, &fixed_value_idx, 1, MPI_LONG_LONG_INT, MPI_MIN, p4est->mpicomm);
    if (fixed_value_idx_g != fixed_value_idx){ // we are not setting the fixed value
      fixed_value_idx_l = -1;
      fixed_value_idx_g = fixed_value_idx;
    } else {
      // reset the value
      ierr = MatZeroRows(A, 1, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
    }
  }

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBase_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

}

void PoissonSolverNodeBase::setup_negative_variable_coeff_laplace_rhsvec()
{
  // register for logging purpose
  ierr = PetscLogEventBegin(log_PoissonSolverNodeBase_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

  double eps = 1E-6*d_min*d_min;
  p4est_topidx_t *t2v = p4est->connectivity->tree_to_vertex;
  double *v2q = p4est->connectivity->vertices;

  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p, *rhs_p, *robin_coef_p = NULL;
  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);

  double *mue_p, *mue_xx_p, *mue_yy_p;
  ierr = VecGetArray(mue_,    &mue_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);

  double *mue_zz_p;
  ierr = VecGetArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(rhs_,    &rhs_p   ); CHKERRXX(ierr);
  if (robin_coef_) { ierr = VecGetArray(robin_coef_, &robin_coef_p); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    p4est_topidx_t tree_it = ni->p.piggy3.which_tree;

    double tree_xmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 0];
    double tree_ymin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 1];
#ifdef P4_TO_P8
    double tree_zmin = v2q[3*t2v[P4EST_CHILDREN*tree_it] + 2];
#endif

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------

    double x_C  = node_x_fr_i(ni) + tree_xmin;
    double y_C  = node_y_fr_j(ni) + tree_ymin;
#ifdef P4_TO_P8
    double z_C  = node_z_fr_k(ni) + tree_zmin;
#endif

    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    double d_m00 = qnnn.d_m00;double d_p00 = qnnn.d_p00;
    double d_0m0 = qnnn.d_0m0;double d_0p0 = qnnn.d_0p0;
#ifdef P4_TO_P8
    double d_00m = qnnn.d_00m;double d_00p = qnnn.d_00p;
#endif

    /*
     * NOTE: All nodes are in PETSc' local numbering
     */
    double d_m00_m0=qnnn.d_m00_m0; double d_m00_p0=qnnn.d_m00_p0;
    double d_p00_m0=qnnn.d_p00_m0; double d_p00_p0=qnnn.d_p00_p0;
    double d_0m0_m0=qnnn.d_0m0_m0; double d_0m0_p0=qnnn.d_0m0_p0;
    double d_0p0_m0=qnnn.d_0p0_m0; double d_0p0_p0=qnnn.d_0p0_p0;
#ifdef P4_TO_P8
    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;
#endif

    if(is_node_Wall(p4est, ni))
    {
#ifdef P4_TO_P8
      if(bc_->wallType(x_C,y_C,z_C) == DIRICHLET)
      {
        rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C);
#else
      if(bc_->wallType(x_C,y_C)     == DIRICHLET)
      {
        rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C);
#endif
        continue;
      }
#ifdef P4_TO_P8
      if(bc_->wallType(x_C,y_C,z_C) == NEUMANN)
#else
      if(bc_->wallType(x_C,y_C)     == NEUMANN)
#endif
      {
        if (is_node_xpWall(p4est, ni)){
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_m00;
#else
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*d_m00;
#endif
          continue;
        }

        if (is_node_xmWall(p4est, ni)){
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_p00;
#else
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*d_p00;
#endif
          continue;
        }

        if (is_node_ypWall(p4est, ni)){
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_0m0;
#else
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*d_0m0;
#endif
          continue;
        }
        if (is_node_ymWall(p4est, ni)){
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_0p0;
#else
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C)*d_0p0;
#endif
          continue;
        }
#ifdef P4_TO_P8
        if (is_node_zpWall(p4est, ni)){
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_00m;
          continue;
        }

        if (is_node_zmWall(p4est, ni)){
          rhs_p[n] = bc_strength*bc_->wallValue(x_C,y_C,z_C)*d_00p;
          continue;
        }
#endif
      }
    } else {
      double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
      double mue_000, mue_p00, mue_m00, mue_0m0, mue_0p0;
#ifdef P4_TO_P8
      double phi_00m, phi_00p;
      double mue_00m, mue_00p;
#endif

#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
      qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
      qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0);
#endif

      //---------------------------------------------------------------------
      // interface boundary
      //---------------------------------------------------------------------
      if((ABS(phi_000)<eps && bc_->interfaceType() == DIRICHLET) ){
#ifdef P4_TO_P8
        rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C,z_C);
#else
        rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C);
#endif
        continue;
      }

      // TODO: This needs optimization
#ifdef P4_TO_P8
      double P_mmm = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min, z_C-0.5*dz_min);
      double P_mpm = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min, z_C-0.5*dz_min);
      double P_pmm = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min, z_C-0.5*dz_min);
      double P_ppm = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min, z_C-0.5*dz_min);
      double P_mmp = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min, z_C+0.5*dz_min);
      double P_mpp = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min, z_C+0.5*dz_min);
      double P_pmp = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min, z_C+0.5*dz_min);
      double P_ppp = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min, z_C+0.5*dz_min);
#else
      double P_mmm = phi_interp(x_C-0.5*dx_min, y_C-0.5*dy_min);
      double P_mpm = phi_interp(x_C-0.5*dx_min, y_C+0.5*dy_min);
      double P_pmm = phi_interp(x_C+0.5*dx_min, y_C-0.5*dy_min);
      double P_ppm = phi_interp(x_C+0.5*dx_min, y_C+0.5*dy_min);
#endif

#ifdef P4_TO_P8
      bool is_one_positive = (P_mmm > 0 || P_pmm > 0 || P_mpm > 0 || P_ppm > 0 ||
                              P_mmp > 0 || P_pmp > 0 || P_mpp > 0 || P_ppp > 0 );
      bool is_one_negative = (P_mmm < 0 || P_pmm < 0 || P_mpm < 0 || P_ppm < 0 ||
                              P_mmp < 0 || P_pmp < 0 || P_mpp < 0 || P_ppp < 0 );

      bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;
#else
      bool is_ngbd_crossed_neumann = ( P_mmm*P_mpm<0 || P_mpm*P_ppm<0 || P_ppm*P_pmm<0 || P_pmm*P_mmm<0 );
#endif

      // far away from the interface
      if(phi_000>0. &&  (!is_ngbd_crossed_neumann || bc_->interfaceType() == DIRICHLET )){
        if(bc_->interfaceType()==DIRICHLET)
#ifdef P4_TO_P8
          rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C,z_C);
#else
          rhs_p[n] = bc_strength*bc_->interfaceValue(x_C,y_C);
#endif
        else
          rhs_p[n] = 0;
        continue;
      }

      // if far away from the interface or close to it but with dirichlet
      // then finite difference method
      if ( (bc_->interfaceType() == DIRICHLET && phi_000<0.) ||
           (bc_->interfaceType() == NEUMANN   && !is_ngbd_crossed_neumann ) ||
           (bc_->interfaceType() == ROBIN     && !is_ngbd_crossed_neumann ) ||
           bc_->interfaceType() == NOINTERFACE)
      {
        double phixx_C = phi_xx_p[n];
        double phiyy_C = phi_yy_p[n];
#ifdef P4_TO_P8
        double phizz_C = phi_zz_p[n];
#endif

        bool is_interface_m00 = (bc_->interfaceType() == DIRICHLET && phi_m00*phi_000 <= 0.);
        bool is_interface_p00 = (bc_->interfaceType() == DIRICHLET && phi_p00*phi_000 <= 0.);
        bool is_interface_0m0 = (bc_->interfaceType() == DIRICHLET && phi_0m0*phi_000 <= 0.);
        bool is_interface_0p0 = (bc_->interfaceType() == DIRICHLET && phi_0p0*phi_000 <= 0.);
#ifdef P4_TO_P8
        bool is_interface_00m = (bc_->interfaceType() == DIRICHLET && phi_00m*phi_000 <= 0.);
        bool is_interface_00p = (bc_->interfaceType() == DIRICHLET && phi_00p*phi_000 <= 0.);
#endif

        double val_interface_m00 = 0.;
        double val_interface_p00 = 0.;
        double val_interface_0m0 = 0.;
        double val_interface_0p0 = 0.;
#ifdef P4_TO_P8
        double val_interface_00m = 0.;
        double val_interface_00p = 0.;
#endif

        // given boundary condition at interface from quadratic interpolation
        if( is_interface_m00) {
          double phixx_m00 = qnnn.f_m00_linear(phi_xx_p);
          double theta_m00 = interface_Location_With_Second_Order_Derivative(0., d_m00, phi_000, phi_m00, phixx_C, phixx_m00);
          if (theta_m00<eps) theta_m00 = eps; if (theta_m00>d_m00) theta_m00 = d_m00;
          d_m00_m0 = d_m00_p0 = 0;
#ifdef P4_TO_P8
          d_m00_0m = d_m00_0p = 0;
#endif

          double mxx_000 = mue_xx_p[n];
          double mxx_m00 = qnnn.f_m00_linear(mue_xx_p);
          mue_m00 = mue_000*(1-theta_m00/d_m00) + mue_m00*theta_m00/d_m00 + 0.5*theta_m00*(theta_m00-d_m00)*MINMOD(mxx_m00,mxx_000);

          d_m00 = theta_m00;

#ifdef P4_TO_P8
          val_interface_m00 = bc_->interfaceValue(x_C - theta_m00, y_C, z_C);
#else
          val_interface_m00 = bc_->interfaceValue(x_C - theta_m00, y_C);
#endif
        }
        if( is_interface_p00){
          double phixx_p00 = qnnn.f_p00_linear(phi_xx_p);
          double theta_p00 = interface_Location_With_Second_Order_Derivative(0., d_p00, phi_000, phi_p00, phixx_C, phixx_p00);
          if (theta_p00<eps) theta_p00 = eps; if (theta_p00>d_p00) theta_p00 = d_p00;
          d_p00_m0 = d_p00_p0 = 0;
#ifdef P4_TO_P8
          d_p00_0m = d_p00_0p = 0;
#endif

          double mxx_000 = mue_xx_p[n];
          double mxx_p00 = qnnn.f_p00_linear(mue_xx_p);
          mue_p00 = mue_000*(1-theta_p00/d_p00) + mue_p00*theta_p00/d_p00 + 0.5*theta_p00*(theta_p00-d_p00)*MINMOD(mxx_p00,mxx_000);

          d_p00 = theta_p00;

#ifdef P4_TO_P8
          val_interface_p00 = bc_->interfaceValue(x_C + theta_p00, y_C, z_C);
#else
          val_interface_p00 = bc_->interfaceValue(x_C + theta_p00, y_C);
#endif
        }
        if( is_interface_0m0){
          double phiyy_0m0 = qnnn.f_0m0_linear(phi_yy_p);
          double theta_0m0 = interface_Location_With_Second_Order_Derivative(0., d_0m0, phi_000, phi_0m0, phiyy_C, phiyy_0m0);
          if (theta_0m0<eps) theta_0m0 = eps; if (theta_0m0>d_0m0) theta_0m0 = d_0m0;
          d_0m0_m0 = d_0m0_p0 = 0;
#ifdef P4_TO_P8
          d_0m0_0m = d_0m0_0p = 0;
#endif

          double myy_000 = mue_yy_p[n];
          double myy_0m0 = qnnn.f_0m0_linear(mue_yy_p);
          mue_0m0 = mue_000*(1-theta_0m0/d_0m0) + mue_0m0*theta_0m0/d_0m0 + 0.5*theta_0m0*(theta_0m0-d_0m0)*MINMOD(myy_0m0,myy_000);

          d_0m0 = theta_0m0;

#ifdef P4_TO_P8
          val_interface_0m0 = bc_->interfaceValue(x_C, y_C - theta_0m0, z_C);
#else
          val_interface_0m0 = bc_->interfaceValue(x_C, y_C - theta_0m0);
#endif
        }
        if( is_interface_0p0){
          double phiyy_0p0 = qnnn.f_0p0_linear(phi_yy_p);
          double theta_0p0 = interface_Location_With_Second_Order_Derivative(0., d_0p0, phi_000, phi_0p0, phiyy_C, phiyy_0p0);
          if (theta_0p0<eps) theta_0p0 = eps; if (theta_0p0>d_0p0) theta_0p0 = d_0p0;
          d_0p0_m0 = d_0p0_p0 = 0;
#ifdef P4_TO_P8

          double myy_000 = mue_yy_p[n];
          double myy_0p0 = qnnn.f_0p0_linear(mue_yy_p);
          mue_0p0 = mue_000*(1-theta_0p0/d_0p0) + mue_0p0*theta_0p0/d_0p0 + 0.5*theta_0p0*(theta_0p0-d_0p0)*MINMOD(myy_0p0,myy_000);

          d_0p0 = theta_0p0;

#endif
          d_0p0 = theta_0p0;
#ifdef P4_TO_P8
          val_interface_0p0 = bc_->interfaceValue(x_C, y_C + theta_0p0, z_C);
#else
          val_interface_0p0 = bc_->interfaceValue(x_C, y_C + theta_0p0);
#endif
        }
#ifdef P4_TO_P8
        if( is_interface_00m){
          double phizz_00m = qnnn.f_00m_linear(phi_zz_p);
          double theta_00m = interface_Location_With_Second_Order_Derivative(0., d_00m, phi_000, phi_00m, phizz_C, phizz_00m);
          if (theta_00m<eps) theta_00m = eps; if (theta_00m>d_00m) theta_00m = d_00m;
          d_00m_m0 = d_00m_p0 = d_00m_0m = d_00m_0p = 0;

          double mzz_000 = mue_zz_p[n];
          double mzz_00m = qnnn.f_00m_linear(mue_zz_p);
          mue_00m = mue_000*(1-theta_00m/d_00m) + mue_00m*theta_00m/d_00m + 0.5*theta_00m*(theta_00m-d_00m)*MINMOD(mzz_00m,mzz_000);

          d_00m = theta_00m;

          val_interface_00m = bc_->interfaceValue(x_C, y_C , z_C - theta_00m);
        }
        if( is_interface_00p){
          double phizz_00p = qnnn.f_00p_linear(phi_zz_p);
          double theta_00p = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000, phi_00p, phizz_C, phizz_00p);
          if (theta_00p<eps) theta_00p = eps; if (theta_00p>d_00p) theta_00p = d_00p;
          d_00p_m0 = d_00p_p0 = d_00p_0m = d_00p_0p = 0;

          double mzz_000 = mue_zz_p[n];
          double mzz_00p = qnnn.f_00p_linear(mue_zz_p);
          mue_00p = mue_000*(1-theta_00p/d_00p) + mue_00p*theta_00p/d_00p + 0.5*theta_00p*(theta_00p-d_00p)*MINMOD(mzz_00p,mzz_000);

          d_00p = theta_00p;

          val_interface_00p = bc_->interfaceValue(x_C, y_C , z_C + theta_00p);
        }
#endif

#ifdef P4_TO_P8
        //------------------------------------
        // Dfxx =   fxx + a*fyy + b*fzz
        // Dfyy = c*fxx +   fyy + d*fzz
        // Dfzz = e*fxx + f*fyy +   fzz
        //------------------------------------
        double a = d_m00_m0*d_m00_p0/d_m00/(d_p00+d_m00) + d_p00_m0*d_p00_p0/d_p00/(d_p00+d_m00) ;
        double b = d_m00_0m*d_m00_0p/d_m00/(d_p00+d_m00) + d_p00_0m*d_p00_0p/d_p00/(d_p00+d_m00) ;

        double c = d_0m0_m0*d_0m0_p0/d_0m0/(d_0p0+d_0m0) + d_0p0_m0*d_0p0_p0/d_0p0/(d_0p0+d_0m0) ;
        double d = d_0m0_0m*d_0m0_0p/d_0m0/(d_0p0+d_0m0) + d_0p0_0m*d_0p0_0p/d_0p0/(d_0p0+d_0m0) ;

        double e = d_00m_m0*d_00m_p0/d_00m/(d_00p+d_00m) + d_00p_m0*d_00p_p0/d_00p/(d_00p+d_00m) ;
        double f = d_00m_0m*d_00m_0p/d_00m/(d_00p+d_00m) + d_00p_0m*d_00p_0p/d_00p/(d_00p+d_00m) ;

        //------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //
        // Laplace = wi*Dfxx +
        //           wj*Dfyy +
        //           wk*Dfzz
        //------------------------------------------------------------

        double det = 1.-a*c-b*e-d*f+a*d*e+b*c*f;
        double wi = (1.-c-e+c*f+e*d-d*f)/det;
        double wj = (1.-a-f+a*e+f*b-b*e)/det;
        double wk = (1.-b-d+b*c+d*a-a*c)/det;
#else
        //------------------------------------------------------------
        // compensating the error of linear interpolation at T-junction using
        // the derivative in the transversal direction
        //
        // Laplace = wi*Dfxx +
        //           wj*Dfyy
        //------------------------------------------------------------

        double wi = 1.0 - d_0m0_m0*d_0m0_p0/d_0m0/(d_0m0+d_0p0) - d_0p0_m0*d_0p0_p0/d_0p0/(d_0m0+d_0p0);
        double wj = 1.0 - d_m00_p0*d_m00_m0/d_m00/(d_m00+d_p00) - d_p00_p0*d_p00_m0/d_p00/(d_m00+d_p00);
#endif

        //---------------------------------------------------------------------
        // Shortley-Weller method, dimension by dimension
        //---------------------------------------------------------------------
        double w_m00 = -(mue_000 + mue_m00)*wi/d_m00/(d_m00+d_p00);
        double w_p00 = -(mue_000 + mue_p00)*wi/d_p00/(d_m00+d_p00);
        double w_0m0 = -(mue_000 + mue_0m0)*wj/d_0m0/(d_0m0+d_0p0);
        double w_0p0 = -(mue_000 + mue_0p0)*wj/d_0p0/(d_0m0+d_0p0);
#ifdef P4_TO_P8
        double w_00m = -(mue_000 + mue_00m)*wk/d_00m/(d_00m+d_00p);
        double w_00p = -(mue_000 + mue_00p)*wk/d_00p/(d_00m+d_00p);
#endif

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
#ifdef P4_TO_P8
        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
#else
        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 );
#endif

        w_m00 /= w_000; w_p00 /= w_000;
        w_0m0 /= w_000; w_0p0 /= w_000;
#ifdef P4_TO_P8
        w_00m /= w_000; w_00p /= w_000;
#endif

        rhs_p[n] /= w_000;

        //---------------------------------------------------------------------
        // add coefficients to the right hand side
        //---------------------------------------------------------------------
        if(is_interface_m00) rhs_p[n] -= w_m00 * val_interface_m00;
        if(is_interface_p00) rhs_p[n] -= w_p00 * val_interface_p00;
        if(is_interface_0m0) rhs_p[n] -= w_0m0 * val_interface_0m0;
        if(is_interface_0p0) rhs_p[n] -= w_0p0 * val_interface_0p0;
#ifdef P4_TO_P8
        if(is_interface_00m) rhs_p[n] -= w_00m * val_interface_00m;
        if(is_interface_00p) rhs_p[n] -= w_00p * val_interface_00p;
#endif
        continue;
      }

      // if ngbd is crossed and neumman BC
      // then use finite volume method
      // only work if the mesh is uniform close to the interface

      // FIXME: the neumann BC on the interface works only if the interface doesn't touch the edge of the domain
      if (is_ngbd_crossed_neumann && (bc_->interfaceType() == NEUMANN || bc_->interfaceType() == ROBIN) )
      {
#ifdef P4_TO_P8
        Cube3 cube;
#else
        Cube2 cube;
#endif
        cube.x0 = x_C-0.5*dx_min;
        cube.x1 = x_C+0.5*dx_min;
        cube.y0 = y_C-0.5*dy_min;
        cube.y1 = y_C+0.5*dy_min;
#ifdef P4_TO_P8
        cube.z0 = z_C-0.5*dz_min;
        cube.z1 = z_C+0.5*dz_min;
#endif
#ifdef P4_TO_P8
        OctValue  phi_cube(P_mmm, P_mmp, P_mpm, P_mpp,
                           P_pmm, P_pmp, P_ppm, P_ppp);
        double volume_cut_cell = cube.volume_In_Negative_Domain(phi_cube);
        double interface_area  = cube.interface_Area_In_Cell(phi_cube);
#else
        QuadValue phi_cube(P_mmm, P_mpm, P_pmm, P_ppm);
        double volume_cut_cell = cube.area_In_Negative_Domain(phi_cube);
        double interface_area  = cube.interface_Length_In_Cell(phi_cube);
#endif

        if (volume_cut_cell>eps*eps)
        {
#ifdef P4_TO_P8
          Cube2 c2;
          QuadValue qv;

          c2.x0    = cube.y0 ; c2.x1    = cube.y1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mmp;    qv.val10 = P_mpm;    qv.val11 = P_mpp;
          double s_m00 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.y0 ; c2.x1    = cube.y1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_pmm;    qv.val01 = P_pmp;    qv.val10 = P_ppm;    qv.val11 = P_ppp;
          double s_p00 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mmp;    qv.val10 = P_pmm;    qv.val11 = P_pmp;
          double s_0m0 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.z0 ; c2.y1    = cube.z1 ;
          qv.val00 = P_mpm;    qv.val01 = P_mpp;    qv.val10 = P_ppm;    qv.val11 = P_ppp;
          double s_0p0 = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.y0 ; c2.y1    = cube.y1 ;
          qv.val00 = P_mmm;    qv.val01 = P_mpm;    qv.val10 = P_pmm;    qv.val11 = P_ppm;
          double s_00m = c2.area_In_Negative_Domain(qv);

          c2.x0    = cube.x0 ; c2.x1    = cube.x1 ; c2.y0    = cube.y0 ; c2.y1    = cube.y1 ;
          qv.val00 = P_mmp;    qv.val01 = P_mpp;    qv.val10 = P_pmp;    qv.val11 = P_ppp;
          double s_00p = c2.area_In_Negative_Domain(qv);

          double w_m00 = -0.5*(mue_000 + mue_m00)*s_m00/dx_min;
          double w_p00 = -0.5*(mue_000 + mue_p00)*s_p00/dx_min;
          double w_0m0 = -0.5*(mue_000 + mue_0m0)*s_0m0/dy_min;
          double w_0p0 = -0.5*(mue_000 + mue_0p0)*s_0p0/dy_min;
          double w_00m = -0.5*(mue_000 + mue_00m)*s_00m/dz_min;
          double w_00p = -0.5*(mue_000 + mue_00p)*s_00p/dz_min;
          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
          if (bc_->interfaceType() == ROBIN){
            if (robin_coef_p[n] > 0) matrix_has_nullspace = false;
            w_000 += robin_coef_p[n]*interface_area;
          }

          OctValue bc_value( bc_->interfaceValue(cube.x0, cube.y0, cube.z0),
                             bc_->interfaceValue(cube.x0, cube.y0, cube.z1),
                             bc_->interfaceValue(cube.x0, cube.y1, cube.z0),
                             bc_->interfaceValue(cube.x0, cube.y1, cube.z1),
                             bc_->interfaceValue(cube.x1, cube.y0, cube.z0),
                             bc_->interfaceValue(cube.x1, cube.y0, cube.z1),
                             bc_->interfaceValue(cube.x1, cube.y1, cube.z0),
                             bc_->interfaceValue(cube.x1, cube.y1, cube.z1));

          double integral_bc = cube.integrate_Over_Interface(bc_value, phi_cube);

          rhs_p[n] *= volume_cut_cell;
          rhs_p[n] += integral_bc;
          rhs_p[n] /= w_000;
#else
          double fxx,fyy;
          fxx = phi_xx_p[n];
          fyy = phi_yy_p[n];

          double s_m00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_mpm, fyy, fyy, dy_min);
          double s_p00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_pmm, P_ppm, fyy, fyy, dy_min);
          double s_0m0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_pmm, fxx, fxx, dx_min);
          double s_0p0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mpm, P_ppm, fxx, fxx, dx_min);

          double w_m00 = -0.5*(mue_000 + mue_m00)*s_m00/dx_min;
          double w_p00 = -0.5*(mue_000 + mue_p00)*s_p00/dx_min;
          double w_0m0 = -0.5*(mue_000 + mue_0m0)*s_0m0/dy_min;
          double w_0p0 = -0.5*(mue_000 + mue_0p0)*s_0p0/dy_min;

          double w_000 = add_p[n]*volume_cut_cell-(w_m00+w_p00+w_0m0+w_0p0);
          if (bc_->interfaceType() == ROBIN){
            if (robin_coef_p[n] > 0) matrix_has_nullspace = false;
            w_000 += robin_coef_p[n]*interface_area;
          }

          QuadValue bc_value( bc_->interfaceValue(cube.x0, cube.y0),
                              bc_->interfaceValue(cube.x0, cube.y1),
                              bc_->interfaceValue(cube.x1, cube.y0),
                              bc_->interfaceValue(cube.x1, cube.y1));

          double integral_bc = cube.integrate_Over_Interface(bc_value, phi_cube);
          rhs_p[n] *= volume_cut_cell;
          rhs_p[n] += integral_bc;
          rhs_p[n] /= w_000;
#endif
        } else {
          rhs_p[n] = 0.;
        }
      }
    }
  }

  if (matrix_has_nullspace && fixed_value_idx_l > 0){
    rhs_p[fixed_value_idx_l] = 0;
  }

  // restore the pointers
  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(mue_,    &mue_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_,    &phi_p   ); CHKERRXX(ierr);
  if (robin_coef_) {
    ierr = VecRestoreArray(robin_coef_, &robin_coef_p); CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_PoissonSolverNodeBase_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void PoissonSolverNodeBase::set_phi(Vec phi, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
void PoissonSolverNodeBase::set_phi(Vec phi, Vec phi_xx, Vec phi_yy)
#endif
{
  phi_ = phi;
  is_matrix_computed = false;

#ifdef P4_TO_P8
  if (phi_xx != NULL && phi_yy != NULL && phi_zz != NULL)
#else
  if (phi_xx != NULL && phi_yy != NULL)
#endif
  {
    phi_xx_ = phi_xx;
    phi_yy_ = phi_yy;
#ifdef P4_TO_P8
    phi_zz_ = phi_zz;
#endif

    is_phi_dd_owned = false;
  } else {

    /*
     * We have two options here:
     * 1) Either compute phi_xx and phi_yy using the function that treats them
     * as two regular functions
     * or,
     * 2) Use the function that uses one block vector and then copy stuff into
     * these two vectors
     *
     * Case 1 requires less communications but case two inccuures additional copies
     * Which one is faster? I don't know!
     *
     * TODO: Going with case 1 for the moment -- to be tested
     */

    // Allocate memory for second derivaties
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz_); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    node_neighbors_->second_derivatives_central(phi_, phi_xx_, phi_yy_, phi_zz_);
#else
    node_neighbors_->second_derivatives_central(phi_, phi_xx_, phi_yy_);
#endif
    is_phi_dd_owned = true;
  }

  // set the interpolating function parameters
#ifdef P4_TO_P8
  phi_interp.set_input_parameters(phi_, quadratic_non_oscillatory, phi_xx_, phi_yy_, phi_zz_);
#else
  phi_interp.set_input_parameters(phi_, quadratic_non_oscillatory, phi_xx_, phi_yy_);
#endif
}

#ifdef P4_TO_P8
void PoissonSolverNodeBase::set_mu(Vec mue, Vec mue_xx, Vec mue_yy, Vec mue_zz)
#else
void PoissonSolverNodeBase::set_mu(Vec mue, Vec mue_xx, Vec mue_yy)
#endif
{
  mue_ = mue;
  is_matrix_computed = false;

#ifdef P4_TO_P8
  if (mue_xx != NULL && mue_yy != NULL && mue_zz != NULL)
#else
  if (mue_xx != NULL && mue_yy != NULL)
#endif
  {
    mue_xx_ = mue_xx;
    mue_yy_ = mue_yy;
#ifdef P4_TO_P8
    mue_zz_ = mue_zz;
#endif

    is_mue_dd_owned = false;
  } else {
    // Allocate memory for second derivaties
    ierr = VecCreateGhostNodes(p4est, nodes, &mue_xx_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &mue_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est, nodes, &mue_zz_); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    node_neighbors_->second_derivatives_central(mue_, mue_xx_, mue_yy_, mue_zz_);
#else
    node_neighbors_->second_derivatives_central(mue_, mue_xx_, mue_yy_);
#endif
    is_mue_dd_owned = true;
  }
}

void PoissonSolverNodeBase::shift_to_exact_solution(Vec sol, Vec uex){
#ifdef CASL_THROWS
  if (!matrix_has_nullspace)
    throw std::runtime_error("[ERROR]: Cannot shift since the original matrix is non-singular");
#endif

  double *uex_p, *sol_p;
  ierr = VecGetArray(uex, &uex_p); CHKERRXX(ierr);
  ierr = VecGetArray(sol, &sol_p); CHKERRXX(ierr);

  int root = -1;
  double shift;

  for (int r = 0; r<p4est->mpisize; r++){
    if (global_node_offset[r] < fixed_value_idx_g && fixed_value_idx_g < global_node_offset[r+1]){
      root = r;
      shift = uex_p[fixed_value_idx_l];

      break;
    }
  }
  MPI_Bcast(&shift, 1, MPI_DOUBLE, root, p4est->mpicomm);

  // shift
  for (size_t i = 0; i<nodes->indep_nodes.elem_count; i++){
    sol_p[i] += shift;
  }

  ierr = VecRestoreArray(uex, &uex_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(sol, &sol_p); CHKERRXX(ierr);
}
