#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_mls.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3_refined_mls.h>
#else
#include "my_p4est_poisson_nodes_mls.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/cube2_refined_mls.h>
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
extern PetscLogEvent log_my_p4est_poisson_nodes_matrix_preallocation;
extern PetscLogEvent log_my_p4est_poisson_nodes_matrix_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_rhsvec_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_KSPSolve;
extern PetscLogEvent log_my_p4est_poisson_nodes_solve;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif
#define bc_strength 1.0

my_p4est_poisson_nodes_mls_t::my_p4est_poisson_nodes_mls_t(const my_p4est_node_neighbors_t *node_neighbors)
  : node_neighbors_(node_neighbors),
    p4est(node_neighbors->p4est), nodes(node_neighbors->nodes), ghost(node_neighbors->ghost), myb_(node_neighbors->myb),
    phi_interp(node_neighbors),
    neumann_wall_first_order(false),
    mu_(1.), diag_add_(0.),
    is_matrix_computed(false), matrix_has_nullspace(false),
    bc_(NULL), A(NULL),
    is_phi_dd_owned(false), is_mue_dd_owned(false),
    rhs_(NULL), phi_(NULL), add_(NULL), mue_(NULL),
    phi_xx_(NULL), phi_yy_(NULL), mue_xx_(NULL), mue_yy_(NULL), robin_coef_(NULL),
    keep_scalling(true), scalling(NULL), phi_eff_(NULL)
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
  p4est_topidx_t vm = p4est->connectivity->tree_to_vertex[0 + 0];
  p4est_topidx_t vp = p4est->connectivity->tree_to_vertex[0 + P4EST_CHILDREN-1];
  double xmin = p4est->connectivity->vertices[3*vm + 0];
  double ymin = p4est->connectivity->vertices[3*vm + 1];
  double xmax = p4est->connectivity->vertices[3*vp + 0];
  double ymax = p4est->connectivity->vertices[3*vp + 1];
  dx_min = (xmax-xmin) / pow(2.,(double) data->max_lvl);
  dy_min = (ymax-ymin) / pow(2.,(double) data->max_lvl);

#ifdef P4_TO_P8
  double zmin = p4est->connectivity->vertices[3*vm + 2];
  double zmax = p4est->connectivity->vertices[3*vp + 2];
  dz_min = (zmax-zmin) / pow(2.,(double) data->max_lvl);
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

  // array to store types of nodes
  node_loc.reserve(nodes->num_owned_indeps);
}

my_p4est_poisson_nodes_mls_t::~my_p4est_poisson_nodes_mls_t()
{
  if (A             != NULL) {ierr = MatDestroy(A);                      CHKERRXX(ierr);}
  if (ksp           != NULL) {ierr = KSPDestroy(ksp);                    CHKERRXX(ierr);}
  if (is_phi_dd_owned){
    if (phi_xx_ != NULL){
      for (int i = 0; i < phi_xx_->size(); i++) {ierr = VecDestroy(phi_xx_->at(i)); CHKERRXX(ierr);}
      delete phi_xx_;
    }

    if (phi_yy_ != NULL){
      for (int i = 0; i < phi_yy_->size(); i++) {ierr = VecDestroy(phi_yy_->at(i)); CHKERRXX(ierr);}
      delete phi_yy_;
    }

#ifdef P4_TO_P8
    if (phi_zz_ != NULL){
      for (int i = 0; i < phi_zz_->size(); i++) {ierr = VecDestroy(phi_zz_->at(i)); CHKERRXX(ierr);}
      delete phi_zz_;
    }
#endif
  }

  if (is_mue_dd_owned){
    if (mue_xx_     != NULL) {ierr = VecDestroy(mue_xx_);                CHKERRXX(ierr);}
    if (mue_yy_     != NULL) {ierr = VecDestroy(mue_yy_);                CHKERRXX(ierr);}
#ifdef P4_TO_P8
    if (mue_zz_     != NULL) {ierr = VecDestroy(mue_zz_);                CHKERRXX(ierr);}
#endif
  }

  if (scalling != NULL) {ierr = VecDestroy(scalling);                CHKERRXX(ierr);}
  if (phi_eff_ != NULL) {ierr = VecDestroy(phi_eff_);                CHKERRXX(ierr);}
}

void my_p4est_poisson_nodes_mls_t::preallocate_matrix()
{  
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

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
  double *phi_eff_p;
  ierr = VecGetArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  for (p4est_locidx_t n=0; n<num_owned_local; n++)
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    /*
     * Check for neighboring nodes:
     * 1) If they exist and are local nodes, increment d_nnz[n]
     * 2) If they exist but are not local nodes, increment o_nnz[n]
     * 3) If they do not exist, simply skip
     */

    if (phi_eff_p[n] > 2*diag_min)
      continue;

//    if (node_loc[n] == NODE_OUT || node_loc[n] == NODE_MXO) continue;

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

  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);

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
//  bool local_phi = false;
//  if(phi_ == NULL)
//  {
//    local_phi = true;
//    ierr = VecDuplicate(solution, &phi_); CHKERRXX(ierr);

//    Vec tmp;
//    ierr = VecGhostGetLocalForm(phi_, &tmp); CHKERRXX(ierr);
//    ierr = VecSet(tmp, -1.); CHKERRXX(ierr);
//    ierr = VecGhostRestoreLocalForm(phi_, &tmp); CHKERRXX(ierr);
////    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi_); CHKERRXX(ierr);
//    set_phi(phi_);
//  }

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
//    if(mue_ == NULL)
//    {
//      if(neumann_wall_first_order)
//        setup_negative_laplace_matrix_neumann_wall_1st_order();
//      else
        setup_negative_laplace_matrix();
//    }
//    else
//      setup_negative_variable_coeff_laplace_matrix();

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

//    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
//    if (matrix_has_nullspace){
//      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
//    }
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // setup rhs
//  if (mue_ == NULL)
//  {
//    if(neumann_wall_first_order)
//      setup_negative_laplace_rhsvec_neumann_wall_1st_order();
//    else
      setup_negative_laplace_rhsvec();
//  }
//  else
//    setup_negative_variable_coeff_laplace_rhsvec();

  // Solve the system
  ierr = KSPSetTolerances(ksp, 1e-14, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_KSPSolve, ksp, rhs_, solution, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs_, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_KSPSolve, ksp, rhs_, solution, 0); CHKERRXX(ierr);

  // update ghosts
  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // get rid of local stuff
  if(local_add)
  {
    ierr = VecDestroy(add_); CHKERRXX(ierr);
    add_ = NULL;
  }
//  if(local_phi)
//  {
//    ierr = VecDestroy(phi_); CHKERRXX(ierr);
//    phi_ = NULL;

//    ierr = VecDestroy(phi_xx_); CHKERRXX(ierr);
//    phi_xx_ = NULL;

//    ierr = VecDestroy(phi_yy_); CHKERRXX(ierr);
//    phi_yy_ = NULL;

//#ifdef P4_TO_P8
//    ierr = VecDestroy(phi_zz_); CHKERRXX(ierr);
//    phi_zz_ = NULL;
//#endif
//  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::setup_negative_laplace_matrix()
{
  preallocate_matrix();

  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

  double eps = 1E-6*d_min*d_min;
//  double eps = 1E-6*d_min;

//  double *phi_p, *phi_xx_p, *phi_yy_p;
//  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
//  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
//  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//  double *phi_zz_p;
//  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
//#endif

  node_vol.resize(nodes->num_owned_indeps, 0);
  int n_phi = phi_->size(); // number of level set functions

  std::vector<double *> phi_p(n_phi, NULL);
  for (int i_phi = 0; i_phi < n_phi; i_phi++) {ierr = VecGetArray((*phi_)[i_phi], &phi_p[i_phi]); CHKERRXX(ierr);}

  double *add_p;
  ierr = VecGetArray(add_, &add_p); CHKERRXX(ierr);

  std::vector<double *> robin_coef_p(n_phi, NULL);

  for (int i = 0; i < n_phi; i++)
  {
    if (robin_coef_ && (*robin_coef_)[i]) {ierr = VecGetArray((*robin_coef_)[i], &robin_coef_p[i]); CHKERRXX(ierr);}
    else                                {robin_coef_p[i] = NULL;}
  }

  std::vector<double> phi_cube(n_phi*P4EST_CHILDREN, -1);

  double *scalling_p;
  if (keep_scalling)
  {
    ierr = VecCreateGhostNodes(p4est, nodes, &scalling); CHKERRXX(ierr);
    ierr = VecGetArray(scalling, &scalling_p); CHKERRXX(ierr);
  }

  double *phi_eff_p; ierr = VecGetArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  // create a cube
#ifdef P4_TO_P8
  cube3_mls_t cube;
  cube3_refined_mls_t cube_refined;
#else
  cube2_mls_t cube;
  cube2_refined_mls_t cube_refined;
#endif

#define CUBE_REFINEMENT 2
  int n_nodes = 1;

  int nx = CUBE_REFINEMENT; std::vector<double> x_coord(nx+1, 0); n_nodes *= (nx+1);
  int ny = CUBE_REFINEMENT; std::vector<double> y_coord(ny+1, 0); n_nodes *= (ny+1);
#ifdef P4_TO_P8
  int nz = CUBE_REFINEMENT; std::vector<double> z_coord(ny+1, 0); n_nodes *= (nz+1);
#endif

  std::vector<double> phi_cube_refined(n_nodes*n_phi, -1);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    if      (phi_eff_p[n] >  4*diag_min)  node_loc[n] = NODE_OUT;
    else if (phi_eff_p[n] < -4*diag_min)  node_loc[n] = NODE_INS;
    else                                  node_loc[n] = NODE_NMN;

    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    double x_C  = node_x_fr_n(n, p4est, nodes);
    double y_C  = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z_C  = node_z_fr_n(n, p4est, nodes);
#endif

    if (node_loc[n] == NODE_NMN)
    {
    cube.x0 = is_node_xmWall(p4est, ni) ? x_C : x_C-0.5*dx_min;
    cube.x1 = is_node_xpWall(p4est, ni) ? x_C : x_C+0.5*dx_min;
    cube.y0 = is_node_ymWall(p4est, ni) ? y_C : y_C-0.5*dy_min;
    cube.y1 = is_node_ypWall(p4est, ni) ? y_C : y_C+0.5*dy_min;

    cube_refined.x0 = is_node_xmWall(p4est, ni) ? x_C : x_C-0.5*dx_min;
    cube_refined.x1 = is_node_xpWall(p4est, ni) ? x_C : x_C+0.5*dx_min;
    cube_refined.y0 = is_node_ymWall(p4est, ni) ? y_C : y_C-0.5*dy_min;
    cube_refined.y1 = is_node_ypWall(p4est, ni) ? y_C : y_C+0.5*dy_min;

#ifdef P4_TO_P8
    cube.z0 = is_node_zmWall(p4est, ni) ? z_C : z_C-0.5*dz_min;
    cube.z1 = is_node_zpWall(p4est, ni) ? z_C : z_C+0.5*dz_min;

    cube_refined.z0 = is_node_zmWall(p4est, ni) ? z_C : z_C-0.5*dz_min;
    cube_refined.z1 = is_node_zpWall(p4est, ni) ? z_C : z_C+0.5*dz_min;
#endif

    // interpolate all level set functions to the cube
    for (int i_phi = 0; i_phi < n_phi; i_phi++){
#ifdef P4_TO_P8
//      phi_interp.set_input(*phi_[i_phi], *phi_xx_[i_phi], *phi_yy_[i_phi], *phi_zz_[i_phi], linear);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = phi_interp(cube.x0, cube.y0, cube.z0);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmp] = phi_interp(cube.x0, cube.y0, cube.z1);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = phi_interp(cube.x0, cube.y1, cube.z0);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpp] = phi_interp(cube.x0, cube.y1, cube.z1);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = phi_interp(cube.x1, cube.y0, cube.z0);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmp] = phi_interp(cube.x1, cube.y0, cube.z1);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = phi_interp(cube.x1, cube.y1, cube.z0);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppp] = phi_interp(cube.x1, cube.y1, cube.z1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y0, cube.z0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmp] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y0, cube.z1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y1, cube.z0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpp] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y1, cube.z1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y0, cube.z0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmp] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y0, cube.z1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y1, cube.z0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppp] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y1, cube.z1);
#else
//        phi_interp.set_input((*phi_)[i_phi], (*phi_xx_)[i_phi], (*phi_yy_)[i_phi], linear);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = phi_interp(cube.x0, cube.y0);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = phi_interp(cube.x0, cube.y1);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = phi_interp(cube.x1, cube.y0);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = phi_interp(cube.x1, cube.y1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y1);
#endif
    }

    // construct geometry inside the cube
    cube.construct_domain(phi_cube.data(), *action_, *color_);

    double dx = (cube.x1 - cube.x0)/(double)(nx); for (int i = 0; i < nx+1; i++) {x_coord[i] = cube.x0 + dx*(double)(i);}
    double dy = (cube.y1 - cube.y0)/(double)(ny); for (int j = 0; j < ny+1; j++) {y_coord[j] = cube.y0 + dy*(double)(j);}
#ifdef P4_TO_P8
    double dz = (cube.z1 - cube.z0)/(double)(nz); for (int k = 0; k < nz+1; k++) {z_coord[k] = cube.z0 + dz*(double)(k);}
#endif

    for (int i_phi = 0; i_phi < n_phi; i_phi++)
      for (int i = 0; i < nx+1; i++)
        for (int j = 0; j < ny+1; j++)
#ifdef P4_TO_P8
          for (int k = 0; k < nz+1; k++)
            phi_cube_refined[i_phi*n_nodes + k*(nx+1)*(ny+1) + j*(nx+1) + i] = (*(*phi_cf_)[i_phi])(x_coord[i], y_coord[j], z_coord[k]);
#else
          phi_cube_refined[i_phi*n_nodes + j*(nx+1) + i] = (*(*phi_cf_)[i_phi])(x_coord[i], y_coord[j]);
#endif

#ifdef P4_TO_P8
    cube_refined.construct_domain(nx, ny, nz, phi_cube_refined.data(), *action_, *color_);
#else
    cube_refined.construct_domain(nx, ny, phi_cube_refined.data(), *action_, *color_);
#endif
    /* end init refined cube */

//    switch (cube.loc){
    switch (cube_refined.loc){
    case INS: node_loc[n] = NODE_INS;  break;
    case FCE: node_loc[n] = NODE_NMN;  break;
    case OUT: node_loc[n] = NODE_OUT;  break;
    }
    }

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

    PetscInt node_000_g = petsc_gloidx[qnnn.node_000];


    std::vector<double> phi_000(n_phi, 0), phi_p00(n_phi, 0), phi_m00(n_phi, 0), phi_0m0(n_phi, 0), phi_0p0(n_phi, 0);
#ifdef P4_TO_P8
    std::vector<double> phi_00m(n_phi, 0), phi_00p(n_phi, 0);
#endif

    for (int i = 0; i < n_phi; i++)
    {
#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i], phi_00m[i], phi_00p[i]);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i]);
#endif
    }

    if(is_node_Wall(p4est, ni) &&
   #ifdef P4_TO_P8
          (*bc_)[0].wallType(x_C,y_C,z_C) == DIRICHLET
   #else
          (*bc_)[0].wallType(x_C,y_C) == DIRICHLET
   #endif
       )
    {
      ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
      if (is_inside(n)) matrix_has_nullspace = false;
      continue;
    } else {
      switch (node_loc[n])
      {
      case NODE_DIR: ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr); break; // no Dirichlet at the moment
      case NODE_OUT: ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr); break;
      case NODE_INS:
      {
        bool is_interface_m00 = false;
        bool is_interface_p00 = false;
        bool is_interface_0m0 = false;
        bool is_interface_0p0 = false;
#ifdef P4_TO_P8
        bool is_interface_00m = false;
        bool is_interface_00p = false;
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
        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0, w_00m=0, w_00p=0;

        if(is_node_xmWall(p4est, ni))      w_p00 += -1./(d_p00*d_p00);
        else if(is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
        else                               w_m00 += -2./d_m00/(d_m00+d_p00);

        if(is_node_xpWall(p4est, ni))      w_m00 += -1./(d_m00*d_m00);
        else if(is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
        else                               w_p00 += -2./d_p00/(d_m00+d_p00);

        if(is_node_ymWall(p4est, ni))      w_0p0 += -1./(d_0p0*d_0p0);
        else if(is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
        else                               w_0m0 += -2./d_0m0/(d_0m0+d_0p0);

        if(is_node_ypWall(p4est, ni))      w_0m0 += -1./(d_0m0*d_0m0);
        else if(is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
        else                               w_0p0 += -2./d_0p0/(d_0m0+d_0p0);

        if(is_node_zmWall(p4est, ni))      w_00p += -1./(d_00p*d_00p);
        else if(is_node_zpWall(p4est, ni)) w_00m += -1./(d_00m*d_00m);
        else                               w_00m += -2./d_00m/(d_00m+d_00p);

        if(is_node_zpWall(p4est, ni))      w_00m += -1./(d_00m*d_00m);
        else if(is_node_zmWall(p4est, ni)) w_00p += -1./(d_00p*d_00p);
        else                               w_00p += -2./d_00p/(d_00m+d_00p);

        w_m00 *= wi * mu_; w_p00 *= wi * mu_;
        w_0m0 *= wj * mu_; w_0p0 *= wj * mu_;
        w_00m *= wk * mu_; w_00p *= wk * mu_;

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
        w_m00 /= w_000; w_p00 /= w_000;
        w_0m0 /= w_000; w_0p0 /= w_000;
        w_00m /= w_000; w_00p /= w_000;

        if (keep_scalling) scalling_p[n] = w_000;

        //---------------------------------------------------------------------
        // add coefficients in the matrix
        //---------------------------------------------------------------------
        if (!is_node_Wall(p4est, ni) && node_000_g < fixed_value_idx_g){
          fixed_value_idx_l = n;
          fixed_value_idx_g = node_000_g;
        }
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
        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;

        if(is_node_xmWall(p4est, ni))      w_p00 += -1./(d_p00*d_p00);
        else if(is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
        else                               w_m00 += -2./d_m00/(d_m00+d_p00);

        if(is_node_xpWall(p4est, ni))      w_m00 += -1./(d_m00*d_m00);
        else if(is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
        else                               w_p00 += -2./d_p00/(d_m00+d_p00);

        if(is_node_ymWall(p4est, ni))      w_0p0 += -1./(d_0p0*d_0p0);
        else if(is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
        else                               w_0m0 += -2./d_0m0/(d_0m0+d_0p0);

        if(is_node_ypWall(p4est, ni))      w_0m0 += -1./(d_0m0*d_0m0);
        else if(is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
        else                               w_0p0 += -2./d_0p0/(d_0m0+d_0p0);

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

        if (keep_scalling) scalling_p[n] = diag;

        //---------------------------------------------------------------------
        // addition to diagonal elements
        //---------------------------------------------------------------------
        if (!is_node_Wall(p4est, ni) && node_000_g < fixed_value_idx_g){
          fixed_value_idx_l = n;
          fixed_value_idx_g = node_000_g;
        }
        ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        if(!is_interface_m00 && !is_node_xmWall(p4est, ni)) {
          PetscInt node_m00_pm_g = petsc_gloidx[node_m00_pm];
          PetscInt node_m00_mm_g = petsc_gloidx[node_m00_mm];

          if (d_m00_m0 != 0) ierr = MatSetValue(A, node_000_g, node_m00_pm_g, w_m00*d_m00_m0/(d_m00_m0+d_m00_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_m00_p0 != 0) ierr = MatSetValue(A, node_000_g, node_m00_mm_g, w_m00*d_m00_p0/(d_m00_m0+d_m00_p0), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_p00 && !is_node_xpWall(p4est, ni)) {
          PetscInt node_p00_pm_g = petsc_gloidx[node_p00_pm];
          PetscInt node_p00_mm_g = petsc_gloidx[node_p00_mm];

          if (d_p00_m0 != 0) ierr = MatSetValue(A, node_000_g, node_p00_pm_g, w_p00*d_p00_m0/(d_p00_m0+d_p00_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_p00_p0 != 0) ierr = MatSetValue(A, node_000_g, node_p00_mm_g, w_p00*d_p00_p0/(d_p00_m0+d_p00_p0), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_0m0 && !is_node_ymWall(p4est, ni)) {
          PetscInt node_0m0_pm_g = petsc_gloidx[node_0m0_pm];
          PetscInt node_0m0_mm_g = petsc_gloidx[node_0m0_mm];

          if (d_0m0_m0 != 0) ierr = MatSetValue(A, node_000_g, node_0m0_pm_g, w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_0m0_p0 != 0) ierr = MatSetValue(A, node_000_g, node_0m0_mm_g, w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0), ADD_VALUES); CHKERRXX(ierr);
        }
        if(!is_interface_0p0 && !is_node_ypWall(p4est, ni)) {
          PetscInt node_0p0_pm_g = petsc_gloidx[node_0p0_pm];
          PetscInt node_0p0_mm_g = petsc_gloidx[node_0p0_mm];

          if (d_0p0_m0 != 0) ierr = MatSetValue(A, node_000_g, node_0p0_pm_g, w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0), ADD_VALUES); CHKERRXX(ierr);
          if (d_0p0_p0 != 0) ierr = MatSetValue(A, node_000_g, node_0p0_mm_g, w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0), ADD_VALUES); CHKERRXX(ierr);
        }
#endif

        if(add_p[n] > 0) matrix_has_nullspace = false;
        continue;

      } break;

      case NODE_NMN:
      {

//        double volume_cut_cell = cube.measure_of_domain();
        double volume_cut_cell = cube_refined.measure_of_domain();

        if (volume_cut_cell < eps*eps)
        {
          ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
          node_loc[n] = NODE_OUT;
        } else {
#ifdef P4_TO_P8
          PetscInt node_m00_g = petsc_gloidx[qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                                              : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp) ];
          PetscInt node_p00_g = petsc_gloidx[qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                                              : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp) ];
          PetscInt node_0m0_g = petsc_gloidx[qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                                              : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp) ];
          PetscInt node_0p0_g = petsc_gloidx[qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                                              : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp) ];
          PetscInt node_00m_g = petsc_gloidx[qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                                              : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp) ];
          PetscInt node_00p_g = petsc_gloidx[qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                                              : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp) ];

//          double s_m00 = cube.measure_in_dir(dir::f_m00);
//          double s_p00 = cube.measure_in_dir(dir::f_p00);
//          double s_0m0 = cube.measure_in_dir(dir::f_0m0);
//          double s_0p0 = cube.measure_in_dir(dir::f_0p0);
//          double s_00m = cube.measure_in_dir(dir::f_00m);
//          double s_00p = cube.measure_in_dir(dir::f_00p);

          double s_m00 = cube_refined.measure_in_dir(dir::f_m00);
          double s_p00 = cube_refined.measure_in_dir(dir::f_p00);
          double s_0m0 = cube_refined.measure_in_dir(dir::f_0m0);
          double s_0p0 = cube_refined.measure_in_dir(dir::f_0p0);
          double s_00m = cube_refined.measure_in_dir(dir::f_00m);
          double s_00p = cube_refined.measure_in_dir(dir::f_00p);

          double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0, w_00m=0, w_00p=0;

          if(!is_node_xmWall(p4est, ni)) w_m00 += -mu_ * s_m00/dx_min;
          if(!is_node_xpWall(p4est, ni)) w_p00 += -mu_ * s_p00/dx_min;
          if(!is_node_ymWall(p4est, ni)) w_0m0 += -mu_ * s_0m0/dy_min;
          if(!is_node_ypWall(p4est, ni)) w_0p0 += -mu_ * s_0p0/dy_min;
          if(!is_node_zmWall(p4est, ni)) w_00m += -mu_ * s_00m/dz_min;
          if(!is_node_zpWall(p4est, ni)) w_00p += -mu_ * s_00p/dz_min;

          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);

          for (int i = 0; i < n_phi; i++)
          {
            if ((*bc_)[i].interfaceType() == ROBIN && robin_coef_p[i] && cube_refined.measure_of_interface(i) > EPS)
            {
              if (fabs(robin_coef_p[i][n]) > 0) matrix_has_nullspace = false;
              if ((*action_)[i] == COLORATION)
              {
                for (int j = 0; j < i; j++)
                {
//                  w_000 += mu_*robin_coef_p[i][n]*cube.measure_of_colored_interface(j,i)/(1.0-1.0*robin_coef_p[i][n]*phi_000[j]);
//                  if (phi_p[j][n] < 0)
//                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_colored_interface(j,i)*(1.0+.0*robin_coef_p[i][n]*phi_p[j][n]);
//                  else
                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_colored_interface(j,i)/(1.0-1.0*robin_coef_p[i][n]*phi_p[j][n]);
                }
              } else {
//                w_000 += mu_*robin_coef_p[i][n]*cube.measure_of_interface(i)/(1.0-1.0*robin_coef_p[i][n]*phi_000[i]);
//                if (phi_p[i][n] < 0)
//                w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_interface(i)*(1.0+.0*robin_coef_p[i][n]*phi_p[i][n]);
//                else
                w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_interface(i)/(1.0-1.0*robin_coef_p[i][n]*phi_p[i][n]);
              }
            }
          }

          //---------------------------------------------------------------------
          // diag scaling
          //---------------------------------------------------------------------
          w_m00 /= w_000; w_p00 /= w_000;
          w_0m0 /= w_000; w_0p0 /= w_000;
          w_00m /= w_000; w_00p /= w_000;

          if (keep_scalling) scalling_p[n] = w_000;

          if (!is_node_Wall(p4est, ni) && node_000_g < fixed_value_idx_g){
            fixed_value_idx_l = n;
            fixed_value_idx_g = node_000_g;
          }
          ierr = MatSetValue(A, node_000_g, node_000_g, 1.0,   ADD_VALUES); CHKERRXX(ierr);
          if(ABS(w_m00) > EPS) {ierr = MatSetValue(A, node_000_g, node_m00_g, w_m00, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_p00) > EPS) {ierr = MatSetValue(A, node_000_g, node_p00_g, w_p00, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_0m0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0m0_g, w_0m0, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_0p0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0p0_g, w_0p0, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_00m) > EPS) {ierr = MatSetValue(A, node_000_g, node_00m_g, w_00m, ADD_VALUES); CHKERRXX(ierr);}
          if(ABS(w_00p) > EPS) {ierr = MatSetValue(A, node_000_g, node_00p_g, w_00p, ADD_VALUES); CHKERRXX(ierr);}

#else
          PetscInt node_m00_g = petsc_gloidx[qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm];
          PetscInt node_p00_g = petsc_gloidx[qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm];
          PetscInt node_0m0_g = petsc_gloidx[qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm];
          PetscInt node_0p0_g = petsc_gloidx[qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm];

//          double s_m00 = cube.measure_in_dir(dir::f_m00);
//          double s_p00 = cube.measure_in_dir(dir::f_p00);
//          double s_0m0 = cube.measure_in_dir(dir::f_0m0);
//          double s_0p0 = cube.measure_in_dir(dir::f_0p0);

          double s_m00 = cube_refined.measure_in_dir(dir::f_m00);
          double s_p00 = cube_refined.measure_in_dir(dir::f_p00);
          double s_0m0 = cube_refined.measure_in_dir(dir::f_0m0);
          double s_0p0 = cube_refined.measure_in_dir(dir::f_0p0);

          double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;
          if(!is_node_xmWall(p4est, ni)) w_m00 += -mu_ * s_m00/dx_min;
          if(!is_node_xpWall(p4est, ni)) w_p00 += -mu_ * s_p00/dx_min;
          if(!is_node_ymWall(p4est, ni)) w_0m0 += -mu_ * s_0m0/dy_min;
          if(!is_node_ypWall(p4est, ni)) w_0p0 += -mu_ * s_0p0/dy_min;

          double w_000 = add_p[n]*volume_cut_cell-(w_m00+w_p00+w_0m0+w_0p0);
//#define ROBIN_COMPLEX
#ifdef ROBIN_COMPLEX
          // determine present interfaces
          std::vector<int> present_interfaces;
          for (int i = 0; i < n_phi; i++)
            if (cube_refined.measure_of_interface(i) > EPS)
              present_interfaces.push_back(i);

          int n_fcs = present_interfaces.size();

          double n1x = 1; double n1y = 0; double alpha1 = 0; double g1 = 0;
          double n2x = 0; double n2y = 1; double alpha2 = 0; double g2 = 0;

          switch (n_fcs)
          {
          case 0: break;
          case 1:
          {
            int i = present_interfaces[0];

            n1x = 0.5*(phi_p00[i]-phi_m00[i])/dx_min;
            n1y = 0.5*(phi_0p0[i]-phi_0m0[i])/dy_min;

            double norm = sqrt(n1x*n1x+n1y*n1y);
            n1x = n1x/norm;
            n1y = n1y/norm;

            alpha1 = robin_coef_p[i][n];
            g1 = (*bc_)[i].interfaceValue(x_C - .0*phi_000[i]*n1x, y_C - .0*phi_000[i]*n1y);

            n2x = -n1y;
            n2y =  n1x;

          } break;
          case 2:
          {
            int i = present_interfaces[0];

            n1x = 0.5*(phi_p00[i]-phi_m00[i])/dx_min;
            n1y = 0.5*(phi_0p0[i]-phi_0m0[i])/dy_min;

            double norm = sqrt(n1x*n1x+n1y*n1y);
            n1x = n1x/norm;
            n1y = n1y/norm;

            alpha1 = robin_coef_p[i][n];
            g1 = (*bc_)[i].interfaceValue(x_C - .0*phi_000[i]*n1x, y_C - .0*phi_000[i]*n1y);

            i = present_interfaces[1];

            n2x = 0.5*(phi_p00[i]-phi_m00[i])/dx_min;
            n2y = 0.5*(phi_0p0[i]-phi_0m0[i])/dy_min;

            norm = sqrt(n2x*n2x+n2y*n2y);
            n2x = n2x/norm;
            n2y = n2y/norm;

            alpha2 = robin_coef_p[i][n];
            g2 = (*bc_)[i].interfaceValue(x_C - .0*phi_000[i]*n2x, y_C - .0*phi_000[i]*n2y);

          } break;
          default: break;
          }

          double denom = (n1x*n2y-n2x*n1y);
          double Ax = (alpha2*n1y-alpha1*n2y)/denom;
          double Ay = (alpha1*n2x-alpha2*n1x)/denom;
          double Bx = (g1*n2y - g2*n1y)/denom;
          double By = (g2*n1x - g1*n2x)/denom;

          double aC0 = Ax;
          double bC0 = Ay;
          double cC0 = 1. - x_C*Ax - y_C*Ay;

          double aC1 = Bx;
          double bC1 = By;
          double cC1 = 0. - x_C*Bx - y_C*By;

          std::vector<double> u_coeff(n_nodes, 0);
          for (int j = 0; j < ny+1; j++)
            for (int i = 0; i < nx+1; i++)
              u_coeff[j*(nx+1)+i] = aC0*x_coord[i] + bC0*y_coord[j] + cC0;

          for (int q = 0; q < n_fcs; q++)
          {
            int i = present_interfaces[q];
            if ((*bc_)[i].interfaceType() == ROBIN && robin_coef_p[i])
            {
              if (fabs(robin_coef_p[i][n]) > 0) matrix_has_nullspace = false;
              if ((*action_)[i] == COLORATION)
              {
                for (int j = 0; j < i; j++)
                {
                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.integrate_over_colored_interface(u_coeff.data(), j,i);
                }
              } else {
                w_000 += mu_*robin_coef_p[i][n]*cube_refined.integrate_over_interface(u_coeff.data(), i);
              }
            }
          }
#else
          for (int i = 0; i < n_phi; i++)
          {
            if ((*bc_)[i].interfaceType() == ROBIN && robin_coef_p[i] && cube_refined.measure_of_interface(i) > EPS)
            {
              if (fabs(robin_coef_p[i][n]) > 0) matrix_has_nullspace = false;
              if ((*action_)[i] == COLORATION)
              {
                for (int j = 0; j < i; j++)
                {
//                  w_000 += mu_*robin_coef_p[i][n]*cube.measure_of_colored_interface(j,i)/(1.0-1.0*robin_coef_p[i][n]*phi_000[j]);
//                  if (phi_p[j][n] < 0)
                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_colored_interface(j,i)*(1.0+1.0*robin_coef_p[i][n]*phi_p[j][n]);
//                  else
//                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_colored_interface(j,i)/(1.0-1.0*robin_coef_p[i][n]*phi_p[j][n]);
                }
              } else {
//                w_000 += mu_*robin_coef_p[i][n]*cube.measure_of_interface(i)/(1.0-1.0*robin_coef_p[i][n]*phi_000[i]);
//                if (phi_p[i][n] < 0)
                w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_interface(i)*(1.0+1.0*robin_coef_p[i][n]*phi_p[i][n]);
//                else
//                w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_interface(i)/(1.0-1.0*robin_coef_p[i][n]*phi_p[i][n]);
              }
            }
          }
#endif

          w_m00 /= w_000; w_p00 /= w_000;
          w_0m0 /= w_000; w_0p0 /= w_000;

          if (keep_scalling) scalling_p[n] = w_000;

          if (!is_node_Wall(p4est, ni) && node_000_g < fixed_value_idx_g){
            fixed_value_idx_l = n;
            fixed_value_idx_g = node_000_g;
          }
          ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
          if (ABS(w_m00) > EPS) {ierr = MatSetValue(A, node_000_g, node_m00_g, w_m00,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_p00) > EPS) {ierr = MatSetValue(A, node_000_g, node_p00_g, w_p00,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0m0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0m0_g, w_0m0,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0p0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0p0_g, w_0p0,  ADD_VALUES); CHKERRXX(ierr);}

#endif

          if(add_p[n] > 0) matrix_has_nullspace = false;

        }
      }
        break;
      }

    }
  }

  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);

  // restore pointers
//  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
//  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
//  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
//#ifdef P4_TO_P8
//  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
//#endif

  ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  ierr = VecRestoreArray(add_, &add_p); CHKERRXX(ierr);

  if (keep_scalling) {ierr = VecRestoreArray(scalling, &scalling_p); CHKERRXX(ierr);}

  for (int i = 0; i < n_phi; i++)
  {
    if (robin_coef_p[i]) {ierr = VecRestoreArray((*robin_coef_)[i], &robin_coef_p[i]);  CHKERRXX(ierr);}
                          ierr = VecRestoreArray((*phi_)[i],        &phi_p[i]);         CHKERRXX(ierr);
  }


  // check for null space
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

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_mls_t::setup_negative_laplace_rhsvec()
{
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

  double eps = 1E-6*d_min*d_min;
//  double eps = 1E-6*d_min;

  double *add_p;  ierr = VecGetArray(add_, &add_p); CHKERRXX(ierr);
  double *rhs_p;  ierr = VecGetArray(rhs_, &rhs_p); CHKERRXX(ierr);
  Vec rhs_dup;
  ierr = VecDuplicate(rhs_, &rhs_dup);  CHKERRXX(ierr);
  ierr = VecCopy(rhs_, rhs_dup);  CHKERRXX(ierr);

  int n_phi = phi_->size(); // number of level set functions

  std::vector<double *> phi_p(n_phi, NULL);
  for (int i_phi = 0; i_phi < n_phi; i_phi++) {ierr = VecGetArray((*phi_)[i_phi], &phi_p[i_phi]); CHKERRXX(ierr);}

  std::vector<double *> robin_coef_p(n_phi, NULL);

  for (int i = 0; i < n_phi; i++)
  {
    if (robin_coef_ && (*robin_coef_)[i]) {ierr = VecGetArray((*robin_coef_)[i], &robin_coef_p[i]); CHKERRXX(ierr);}
    else                                {robin_coef_p[i] = NULL;}
  }

  std::vector<double> phi_cube(n_phi*P4EST_CHILDREN, -1);

  double *scalling_p;
  if (keep_scalling)
  {
    ierr = VecGetArray(scalling, &scalling_p); CHKERRXX(ierr);
  }

  int n_nodes = 1;

  int nx = CUBE_REFINEMENT;   std::vector<double> x_coord(nx+1, 0);   n_nodes *= (nx+1);
  int ny = CUBE_REFINEMENT;   std::vector<double> y_coord(ny+1, 0);   n_nodes *= (ny+1);
#ifdef P4_TO_P8
  int nz = CUBE_REFINEMENT;   std::vector<double> z_coord(nz+1, 0);   n_nodes *= (nz+1);
#endif

  std::vector<double> phi_cube_refined(n_nodes*n_phi, -1);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    double x_C  = node_x_fr_n(n, p4est, nodes);
    double y_C  = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
    double z_C  = node_z_fr_n(n, p4est, nodes);
#endif

    // create a cube
#ifdef P4_TO_P8
    cube3_mls_t cube;
    cube3_refined_mls_t cube_refined;
#else
    cube2_mls_t cube;
    cube2_refined_mls_t cube_refined;
#endif

    if (node_loc[n] == NODE_NMN)
    {
    cube.x0 = is_node_xmWall(p4est, ni) ? x_C : x_C-0.5*dx_min;
    cube.x1 = is_node_xpWall(p4est, ni) ? x_C : x_C+0.5*dx_min;
    cube.y0 = is_node_ymWall(p4est, ni) ? y_C : y_C-0.5*dy_min;
    cube.y1 = is_node_ypWall(p4est, ni) ? y_C : y_C+0.5*dy_min;
#ifdef P4_TO_P8
    cube.z0 = is_node_zmWall(p4est, ni) ? z_C : z_C-0.5*dz_min;
    cube.z1 = is_node_zpWall(p4est, ni) ? z_C : z_C+0.5*dz_min;
#endif

    cube_refined.x0 = is_node_xmWall(p4est, ni) ? x_C : x_C-0.5*dx_min;
    cube_refined.x1 = is_node_xpWall(p4est, ni) ? x_C : x_C+0.5*dx_min;
    cube_refined.y0 = is_node_ymWall(p4est, ni) ? y_C : y_C-0.5*dy_min;
    cube_refined.y1 = is_node_ypWall(p4est, ni) ? y_C : y_C+0.5*dy_min;
#ifdef P4_TO_P8
    cube_refined.z0 = is_node_zmWall(p4est, ni) ? z_C : z_C-0.5*dz_min;
    cube_refined.z1 = is_node_zpWall(p4est, ni) ? z_C : z_C+0.5*dz_min;
#endif

    // interpolate all level set functions to the cube
    for (int i_phi = 0; i_phi < n_phi; i_phi++){
#ifdef P4_TO_P8
//      phi_interp.set_input(*phi_[i_phi], *phi_xx_[i_phi], *phi_yy_[i_phi], *phi_zz_[i_phi], linear);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = phi_interp(cube.x0, cube.y0, cube.z0);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmp] = phi_interp(cube.x0, cube.y0, cube.z1);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = phi_interp(cube.x0, cube.y1, cube.z0);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpp] = phi_interp(cube.x0, cube.y1, cube.z1);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = phi_interp(cube.x1, cube.y0, cube.z0);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmp] = phi_interp(cube.x1, cube.y0, cube.z1);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = phi_interp(cube.x1, cube.y1, cube.z0);
//      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppp] = phi_interp(cube.x1, cube.y1, cube.z1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y0, cube.z0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmp] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y0, cube.z1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y1, cube.z0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpp] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y1, cube.z1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y0, cube.z0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmp] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y0, cube.z1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y1, cube.z0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppp] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y1, cube.z1);
#else
//        phi_interp.set_input((*phi_)[i_phi], (*phi_xx_)[i_phi], (*phi_yy_)[i_phi], linear);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = phi_interp(cube.x0, cube.y0);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = phi_interp(cube.x0, cube.y1);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = phi_interp(cube.x1, cube.y0);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = phi_interp(cube.x1, cube.y1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = (*(*phi_cf_)[i_phi])(cube.x0, cube.y1);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y0);
      phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = (*(*phi_cf_)[i_phi])(cube.x1, cube.y1);
#endif
    }

    cube.construct_domain(phi_cube.data(), *action_, *color_);

    double dx = (cube.x1 - cube.x0)/(double)(nx);   for (int i = 0; i < nx+1; i++) {x_coord[i] = cube.x0 + dx*(double)(i);}
    double dy = (cube.y1 - cube.y0)/(double)(ny);   for (int j = 0; j < ny+1; j++) {y_coord[j] = cube.y0 + dy*(double)(j);}
#ifdef P4_TO_P8
    double dz = (cube.z1 - cube.z0)/(double)(nz);   for (int k = 0; k < nz+1; k++) {z_coord[k] = cube.z0 + dz*(double)(k);}
#endif

    for (int i_phi = 0; i_phi < n_phi; i_phi++)
      for (int i = 0; i < nx+1; i++)
        for (int j = 0; j < ny+1; j++)
#ifdef P4_TO_P8
          for (int k = 0; k < nz+1; k++)
            phi_cube_refined[i_phi*n_nodes + k*(nx+1)*(ny+1) + j*(nx+1) + i] = (*(*phi_cf_)[i_phi])(x_coord[i], y_coord[j], z_coord[k]);
#else
          phi_cube_refined[i_phi*n_nodes + j*(nx+1) + i] = (*(*phi_cf_)[i_phi])(x_coord[i], y_coord[j]);
#endif

#ifdef P4_TO_P8
    cube_refined.construct_domain(nx, ny, nz, phi_cube_refined.data(), *action_, *color_);
#else
    cube_refined.construct_domain(nx, ny, phi_cube_refined.data(), *action_, *color_);
#endif
    }

    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    std::vector<double> phi_000(n_phi, 0), phi_p00(n_phi, 0), phi_m00(n_phi, 0), phi_0m0(n_phi, 0), phi_0p0(n_phi, 0);
#ifdef P4_TO_P8
    std::vector<double> phi_00m(n_phi, 0), phi_00p(n_phi, 0);
#endif

    for (int i = 0; i < n_phi; i++)
    {
#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i], phi_00m[i], phi_00p[i]);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p[i], phi_000[i], phi_m00[i], phi_p00[i], phi_0m0[i], phi_0p0[i]);
#endif
    }

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

    if(is_node_Wall(p4est, ni)
       &&
#ifdef P4_TO_P8
       (*bc_)[0].wallType(x_C,y_C,z_C) == DIRICHLET
#else
       (*bc_)[0].wallType(x_C,y_C) == DIRICHLET
#endif
       )
    {
#ifdef P4_TO_P8
      rhs_p[n] = bc_strength*(*bc_)[0].wallValue(x_C,y_C,z_C);
#else
      rhs_p[n] = bc_strength*(*bc_)[0].wallValue(x_C,y_C);
#endif
      continue;
    }
    else
    {
      switch (node_loc[n])
      {
      case NODE_DIR: rhs_p[n] = 0; break; // no DIRICHLET BC at the moment
      case NODE_OUT: rhs_p[n] = 0; break;
      case NODE_INS:
      {
#ifdef P4_TO_P8
        double diag;
        if (!keep_scalling)
        {
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
          double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0, w_00m=0, w_00p=0;

          if(is_node_xmWall(p4est, ni))      w_p00 += -1./(d_p00*d_p00);
          else if(is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
          else                               w_m00 += -2./d_m00/(d_m00+d_p00);

          if(is_node_xpWall(p4est, ni))      w_m00 += -1./(d_m00*d_m00);
          else if(is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
          else                               w_p00 += -2./d_p00/(d_m00+d_p00);

          if(is_node_ymWall(p4est, ni))      w_0p0 += -1./(d_0p0*d_0p0);
          else if(is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
          else                               w_0m0 += -2./d_0m0/(d_0m0+d_0p0);

          if(is_node_ypWall(p4est, ni))      w_0m0 += -1./(d_0m0*d_0m0);
          else if(is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
          else                               w_0p0 += -2./d_0p0/(d_0m0+d_0p0);

          if(is_node_zmWall(p4est, ni))      w_00p += -1./(d_00p*d_00p);
          else if(is_node_zpWall(p4est, ni)) w_00m += -1./(d_00m*d_00m);
          else                               w_00m += -2./d_00m/(d_00m+d_00p);

          if(is_node_zpWall(p4est, ni))      w_00m += -1./(d_00m*d_00m);
          else if(is_node_zmWall(p4est, ni)) w_00p += -1./(d_00p*d_00p);
          else                               w_00p += -2./d_00p/(d_00m+d_00p);

          w_m00 *= wi * mu_; w_p00 *= wi * mu_;
          w_0m0 *= wj * mu_; w_0p0 *= wj * mu_;
          w_00m *= wk * mu_; w_00p *= wk * mu_;

          //---------------------------------------------------------------------
          // diag scaling
          //---------------------------------------------------------------------
          diag = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
        } else {
          diag = scalling_p[n];
        }
        rhs_p[n] /= diag;
#else
        double diag;
        if (!keep_scalling)
        {
          //---------------------------------------------------------------------
          // Shortley-Weller method, dimension by dimension
          //---------------------------------------------------------------------
          double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;

          if(is_node_xmWall(p4est, ni))      w_p00 += -1./(d_p00*d_p00);
          else if(is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
          else                               w_m00 += -2./d_m00/(d_m00+d_p00);

          if(is_node_xpWall(p4est, ni))      w_m00 += -1./(d_m00*d_m00);
          else if(is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
          else                               w_p00 += -2./d_p00/(d_m00+d_p00);

          if(is_node_ymWall(p4est, ni))      w_0p0 += -1./(d_0p0*d_0p0);
          else if(is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
          else                               w_0m0 += -2./d_0m0/(d_0m0+d_0p0);

          if(is_node_ypWall(p4est, ni))      w_0m0 += -1./(d_0m0*d_0m0);
          else if(is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
          else                               w_0p0 += -2./d_0p0/(d_0m0+d_0p0);

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

          diag = add_p[n]-(w_m00+w_p00+w_0m0+w_0p0);
        } else {
          diag = scalling_p[n];
        }

        rhs_p[n] /= diag;
#endif
      } break;

      case NODE_NMN:
      {
//        double volume_cut_cell = cube.measure_of_domain();
        double volume_cut_cell = cube_refined.measure_of_domain();

        if (volume_cut_cell < eps*eps)
        {
          rhs_p[n] = 0.;
        } else {
#ifdef P4_TO_P8
          double w_000;
          if (!keep_scalling)
          {
//            double s_m00 = cube.measure_in_dir(dir::f_m00);
//            double s_p00 = cube.measure_in_dir(dir::f_p00);
//            double s_0m0 = cube.measure_in_dir(dir::f_0m0);
//            double s_0p0 = cube.measure_in_dir(dir::f_0p0);
//            double s_00m = cube.measure_in_dir(dir::f_00m);
//            double s_00p = cube.measure_in_dir(dir::f_00p);

            double s_m00 = cube_refined.measure_in_dir(dir::f_m00);
            double s_p00 = cube_refined.measure_in_dir(dir::f_p00);
            double s_0m0 = cube_refined.measure_in_dir(dir::f_0m0);
            double s_0p0 = cube_refined.measure_in_dir(dir::f_0p0);
            double s_00m = cube_refined.measure_in_dir(dir::f_00m);
            double s_00p = cube_refined.measure_in_dir(dir::f_00p);

            double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0, w_00m=0, w_00p=0;

            if(!is_node_xmWall(p4est, ni)) w_m00 += -mu_ * s_m00/dx_min;
            if(!is_node_xpWall(p4est, ni)) w_p00 += -mu_ * s_p00/dx_min;
            if(!is_node_ymWall(p4est, ni)) w_0m0 += -mu_ * s_0m0/dy_min;
            if(!is_node_ypWall(p4est, ni)) w_0p0 += -mu_ * s_0p0/dy_min;
            if(!is_node_zmWall(p4est, ni)) w_00m += -mu_ * s_00m/dz_min;
            if(!is_node_zpWall(p4est, ni)) w_00p += -mu_ * s_00p/dz_min;

            w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);

            for (int i = 0; i < n_phi; i++)
            {
              if ((*bc_)[i].interfaceType() == ROBIN && robin_coef_p[i] && cube_refined.measure_of_interface(i) > EPS)
              {
                if (fabs(robin_coef_p[i][n]) > 0) matrix_has_nullspace = false;
                if ((*action_)[i] == COLORATION)
                {
                  for (int j = 0; j < i; j++)
                  {
//                    w_000 += mu_*robin_coef_p[i][n]*cube.measure_of_colored_interface(j,i)/(1.0-1.0*robin_coef_p[i][n]*phi_000[j]);
//                    if (phi_p[j][n] < 0)
//                    w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_colored_interface(j,i)*(1.0+1.0*robin_coef_p[i][n]*phi_p[j][n]);
//                    else
                    w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_colored_interface(j,i)/(1.0-0.0*robin_coef_p[i][n]*phi_p[j][n]);
                  }
                } else {
//                  w_000 += mu_*robin_coef_p[i][n]*cube.measure_of_interface(i)/(1.0-1.0*robin_coef_p[i][n]*phi_000[i]);
//                  if (phi_p[i][n] < 0)
//                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_interface(i)*(1.0+1.0*robin_coef_p[i][n]*phi_p[i][n]);
//                  else
                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_interface(i)/(1.0-0.0*robin_coef_p[i][n]*phi_p[i][n]);
                }
              }
            }
          } else {
            w_000 = scalling_p[n];
          }

          double bc_value[P4EST_CHILDREN];
          std::vector<double> bc_value_refined(n_nodes, 0);

//          for (int ii = 0; ii < nx+1; ii++)
//            for (int jj = 0; jj < ny+1; jj++)
//              for (int kk = 0; kk < nz+1; kk++)
//                bc_value_refined[kk*(nx+1)*(ny+1) + jj*(nx+1) + ii] = (*force_)(x_coord[ii], y_coord[jj], z_coord[kk]);

//          rhs_p[n] = cube_refined.integrate_over_domain(bc_value_refined.data());

          rhs_p[n] *= volume_cut_cell;

          for (int i = 0; i < n_phi; i++)
          {
            if (cube_refined.measure_of_interface(i) > EPS)
            {
              double dpdx = 0.5*(phi_p00[i]-phi_m00[i])/dx_min;
              double dpdy = 0.5*(phi_0p0[i]-phi_0m0[i])/dy_min;
              double dpdz = 0.5*(phi_00p[i]-phi_00m[i])/dz_min;
              double norm = sqrt(dpdx*dpdx+dpdy*dpdy+dpdz*dpdz);
              double dist = 1.0*phi_000[i];
              double g_add;
//            if (phi_p[i][n] < 0)
//              g_add = 0.0*robin_coef_p[i][n]*dist*(*bc_)[i].interfaceValue(x_C - 1.0*dist*dpdx/norm, y_C - 1.0*dist*dpdy/norm, z_C - 1.0*dist*dpdz/norm)*(1.0+robin_coef_p[i][n]*dist);
//            else
            g_add = 1.0*robin_coef_p[i][n]*dist*(*bc_)[i].interfaceValue(x_C - 1.0*dist*dpdx/norm, y_C - 1.0*dist*dpdy/norm, z_C - 1.0*dist*dpdz/norm)/(1.0-robin_coef_p[i][n]*dist);
//            double g_add = 1.0*robin_coef_p[i][n]*dist*bc_avg/(1.0-robin_coef_p[i][n]*dist);

              bc_value[0] = (*bc_)[i].interfaceValue(cube.x0, cube.y0, cube.z0)+g_add;
              bc_value[1] = (*bc_)[i].interfaceValue(cube.x1, cube.y0, cube.z0)+g_add;
              bc_value[2] = (*bc_)[i].interfaceValue(cube.x0, cube.y1, cube.z0)+g_add;
              bc_value[3] = (*bc_)[i].interfaceValue(cube.x1, cube.y1, cube.z0)+g_add;
              bc_value[4] = (*bc_)[i].interfaceValue(cube.x0, cube.y0, cube.z1)+g_add;
              bc_value[5] = (*bc_)[i].interfaceValue(cube.x1, cube.y0, cube.z1)+g_add;
              bc_value[6] = (*bc_)[i].interfaceValue(cube.x0, cube.y1, cube.z1)+g_add;
              bc_value[7] = (*bc_)[i].interfaceValue(cube.x1, cube.y1, cube.z1)+g_add;

              double integral_bc = 0;


              for (int ii = 0; ii < nx+1; ii++)
                for (int jj = 0; jj < ny+1; jj++)
                  for (int kk = 0; kk < nz+1; kk++)
                    bc_value_refined[kk*(nx+1)*(ny+1) + jj*(nx+1) + ii] = (*bc_)[i].interfaceValue(x_coord[ii], y_coord[jj], z_coord[kk])+g_add;

              if ((*action_)[i] == COLORATION)
              {
                for (int j = 0; j < i; j++)
                {
//                integral_bc += cube.integrate_over_colored_interface(bc_value, j, i)/(1.0-0.0*robin_coef_p[i][n]*phi_000[j]);
//                if (phi_p[j][n] < 0)
//                  integral_bc += cube_refined.integrate_over_colored_interface(bc_value_refined.data(), j, i)*(1.0+.0*robin_coef_p[i][n]*phi_p[j][n]);
//                else
                integral_bc += cube_refined.integrate_over_colored_interface(bc_value_refined.data(), j, i)/(1.0-.0*robin_coef_p[i][n]*phi_p[j][n]);
                }
              } else {
//              integral_bc = cube.integrate_over_interface(bc_value, i)/(1.0-0.0*robin_coef_p[i][n]*phi_000[i]);
//              if (phi_p[i][n] < 0)
//                integral_bc = cube_refined.integrate_over_interface(bc_value_refined.data(), i)*(1.0+.0*robin_coef_p[i][n]*phi_p[i][n]);
//              else
              integral_bc = cube_refined.integrate_over_interface(bc_value_refined.data(), i)/(1.0-.0*robin_coef_p[i][n]*phi_p[i][n]);
              }

              rhs_p[n] += mu_*integral_bc;
            }
          }

//          if (is_node_xmWall(p4est, ni)) rhs_p[n] += mu_*s_m00*bc_->wallValue(x_C, y_C, z_C);
//          if (is_node_xpWall(p4est, ni)) rhs_p[n] += mu_*s_p00*bc_->wallValue(x_C, y_C, z_C);
//          if (is_node_ymWall(p4est, ni)) rhs_p[n] += mu_*s_0m0*bc_->wallValue(x_C, y_C, z_C);
//          if (is_node_ypWall(p4est, ni)) rhs_p[n] += mu_*s_0p0*bc_->wallValue(x_C, y_C, z_C);
//          if (is_node_zmWall(p4est, ni)) rhs_p[n] += mu_*s_00m*bc_->wallValue(x_C, y_C, z_C);
//          if (is_node_zpWall(p4est, ni)) rhs_p[n] += mu_*s_00p*bc_->wallValue(x_C, y_C, z_C);

          rhs_p[n] /= w_000;
#else
          double w_000;
//          if (!keep_scalling)
//          {
//            double s_m00 = cube.measure_in_dir(dir::f_m00);
//            double s_p00 = cube.measure_in_dir(dir::f_p00);
//            double s_0m0 = cube.measure_in_dir(dir::f_0m0);
//            double s_0p0 = cube.measure_in_dir(dir::f_0p0);

            double s_m00 = cube_refined.measure_in_dir(dir::f_m00);
            double s_p00 = cube_refined.measure_in_dir(dir::f_p00);
            double s_0m0 = cube_refined.measure_in_dir(dir::f_0m0);
            double s_0p0 = cube_refined.measure_in_dir(dir::f_0p0);

            double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;
            if(!is_node_xmWall(p4est, ni)) w_m00 += -mu_ * s_m00/dx_min;
            if(!is_node_xpWall(p4est, ni)) w_p00 += -mu_ * s_p00/dx_min;
            if(!is_node_ymWall(p4est, ni)) w_0m0 += -mu_ * s_0m0/dy_min;
            if(!is_node_ypWall(p4est, ni)) w_0p0 += -mu_ * s_0p0/dy_min;

            w_000 = add_p[n]*volume_cut_cell-(w_m00+w_p00+w_0m0+w_0p0);

#ifdef ROBIN_COMPLEX
            // determine present interfaces
            std::vector<int> present_interfaces;
            for (int i = 0; i < n_phi; i++)
              if (cube_refined.measure_of_interface(i) > eps)
                present_interfaces.push_back(i);

            int n_fcs = present_interfaces.size();

            double n1x = 1; double n1y = 0; double alpha1 = 0; double g1 = 0;
            double n2x = 0; double n2y = 1; double alpha2 = 0; double g2 = 0;

            switch (n_fcs)
            {
            case 0: break;
            case 1:
            {
              int i = present_interfaces[0];

              n1x = 0.5*(phi_p00[i]-phi_m00[i])/dx_min;
              n1y = 0.5*(phi_0p0[i]-phi_0m0[i])/dy_min;

              double norm = sqrt(n1x*n1x+n1y*n1y);
              n1x = n1x/norm;
              n1y = n1y/norm;

              alpha1 = robin_coef_p[i][n];
              g1 = (*bc_)[i].interfaceValue(x_C - 1.0*phi_000[i]*n1x, y_C - 1.0*phi_000[i]*n1y);

              n2x = -n1y;
              n2y =  n1x;

            } break;
            case 2:
            {
              int i = present_interfaces[0];

              n1x = 0.5*(phi_p00[i]-phi_m00[i])/dx_min;
              n1y = 0.5*(phi_0p0[i]-phi_0m0[i])/dy_min;

              double norm = sqrt(n1x*n1x+n1y*n1y);
              n1x = n1x/norm;
              n1y = n1y/norm;

              alpha1 = robin_coef_p[i][n];
              g1 = (*bc_)[i].interfaceValue(x_C - 1.0*phi_000[i]*n1x, y_C - 1.0*phi_000[i]*n1y);

              i = present_interfaces[1];

              n2x = 0.5*(phi_p00[i]-phi_m00[i])/dx_min;
              n2y = 0.5*(phi_0p0[i]-phi_0m0[i])/dy_min;

              norm = sqrt(n2x*n2x+n2y*n2y);
              n2x = n2x/norm;
              n2y = n2y/norm;

              alpha2 = robin_coef_p[i][n];
              g2 = (*bc_)[i].interfaceValue(x_C - 1.0*phi_000[i]*n2x, y_C - 1.0*phi_000[i]*n2y);

            } break;
            default:
              throw std::domain_error("[CASL_ERROR]:"); break;
            }

            double denom = (n1x*n2y-n2x*n1y);
            double Ax = (alpha2*n1y-alpha1*n2y)/denom;
            double Ay = (alpha1*n2x-alpha2*n1x)/denom;
            double Bx = (g1*n2y - g2*n1y)/denom;
            double By = (g2*n1x - g1*n2x)/denom;

            double aC0 = Ax;
            double bC0 = Ay;
            double cC0 = 1. - x_C*Ax - y_C*Ay;

            double aC1 = Bx;
            double bC1 = By;
            double cC1 = 0. - x_C*Bx - y_C*By;

            std::vector<double> u_coeff(n_nodes, 0);
            for (int j = 0; j < ny+1; j++)
              for (int i = 0; i < nx+1; i++)
                u_coeff[j*(nx+1)+i] = aC0*x_coord[i] + bC0*y_coord[j] + cC0;

            for (int q = 0; q < n_fcs; q++)
            {
              int i = present_interfaces[q];
              if ((*bc_)[i].interfaceType() == ROBIN && robin_coef_p[i])
              {
                if (fabs(robin_coef_p[i][n]) > 0) matrix_has_nullspace = false;
                if ((*action_)[i] == COLORATION)
                {
                  for (int j = 0; j < i; j++)
                  {
                    w_000 += mu_*robin_coef_p[i][n]*cube_refined.integrate_over_colored_interface(u_coeff.data(), j,i);
                  }
                } else {
                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.integrate_over_interface(u_coeff.data(), i);
                }
              }
            }
#else
            for (int i = 0; i < n_phi; i++)
            {
              if ((*bc_)[i].interfaceType() == ROBIN && robin_coef_p[i] && cube_refined.measure_of_interface(i) > EPS)
              {
                if (fabs(robin_coef_p[i][n]) > 0) matrix_has_nullspace = false;
                if ((*action_)[i] == COLORATION)
                {
                  for (int j = 0; j < i; j++)
                  {
//                    w_000 += mu_*robin_coef_p[i][n]*cube.measure_of_colored_interface(j,i)/(1.0-1.0*robin_coef_p[i][n]*phi_000[j]);
//                    if (phi_p[j][n] < 0)
                    w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_colored_interface(j,i)*(1.0+1.0*robin_coef_p[i][n]*phi_p[j][n]);
//                    else
//                    w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_colored_interface(j,i)/(1.0-1.0*robin_coef_p[i][n]*phi_p[j][n]);
                  }
                } else {
//                  w_000 += mu_*robin_coef_p[i][n]*cube.measure_of_interface(i)/(1.0-1.0*robin_coef_p[i][n]*phi_000[i]);
//                  if (phi_p[i][n] < 0)
                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_interface(i)*(1.0+1.0*robin_coef_p[i][n]*phi_p[i][n]);
//                  else
//                  w_000 += mu_*robin_coef_p[i][n]*cube_refined.measure_of_interface(i)/(1.0-1.0*robin_coef_p[i][n]*phi_p[i][n]);
                }
              }
            }
#endif
//          } else {
            w_000 = scalling_p[n];
//          }

          double bc_value[P4EST_CHILDREN];

          std::vector<double> bc_value_refined(n_nodes, 0);

          rhs_p[n] *= volume_cut_cell;

          for (int i = 0; i < n_phi; i++)
          {
            if (cube_refined.measure_of_interface(i) > EPS)
            {
            for (int ii = 0; ii < nx+1; ii++)
              for (int jj = 0; jj < ny+1; jj++)
                bc_value_refined[jj*(nx+1) + ii] = (*bc_)[i].interfaceValue(x_coord[ii], y_coord[jj]);

            double bc_avg = cube_refined.integrate_over_interface(bc_value_refined.data(), i)/cube_refined.measure_of_interface(i);

            double dpdx = 0.5*(phi_p00[i]-phi_m00[i])/dx_min;
            double dpdy = 0.5*(phi_0p0[i]-phi_0m0[i])/dy_min;
            double norm = sqrt(dpdx*dpdx+dpdy*dpdy);
            double dist = 1.0*phi_000[i];
            double g_add;
//            if (phi_p[i][n] < 0)
            g_add = 1.0*robin_coef_p[i][n]*dist*(*bc_)[i].interfaceValue(x_C - .0*dist*dpdx/norm, y_C - .0*dist*dpdy/norm)*(1.0+robin_coef_p[i][n]*dist);
//            else
//            g_add = 1.0*robin_coef_p[i][n]*dist*(*bc_)[i].interfaceValue(x_C - .0*dist*dpdx/norm, y_C - .0*dist*dpdy/norm)/(1.0-robin_coef_p[i][n]*dist);
//            double g_add = 1.0*robin_coef_p[i][n]*dist*bc_avg/(1.0-robin_coef_p[i][n]*dist);

            bc_value[0] = (*bc_)[i].interfaceValue(cube.x0, cube.y0)+g_add;;
            bc_value[1] = (*bc_)[i].interfaceValue(cube.x1, cube.y0)+g_add;;
            bc_value[2] = (*bc_)[i].interfaceValue(cube.x0, cube.y1)+g_add;;
            bc_value[3] = (*bc_)[i].interfaceValue(cube.x1, cube.y1)+g_add;;

            double integral_bc = 0;
#ifdef ROBIN_COMPLEX
            for (int ii = 0; ii < nx+1; ii++)
              for (int jj = 0; jj < ny+1; jj++)
                bc_value_refined[jj*(nx+1) + ii] = (*bc_)[i].interfaceValue(x_coord[ii], y_coord[jj]) - robin_coef_p[i][n]*(aC1*x_coord[ii]+bC1*y_coord[jj]+cC1);


            if ((*action_)[i] == COLORATION)
            {
              for (int j = 0; j < i; j++)
              {
                integral_bc += cube_refined.integrate_over_colored_interface(bc_value_refined.data(), j, i);
              }
            } else {
              integral_bc = cube_refined.integrate_over_interface(bc_value_refined.data(), i);
            }
#else
            for (int ii = 0; ii < nx+1; ii++)
              for (int jj = 0; jj < ny+1; jj++)
                bc_value_refined[jj*(nx+1) + ii] = (*bc_)[i].interfaceValue(x_coord[ii], y_coord[jj])+g_add;

            if ((*action_)[i] == COLORATION)
            {
              for (int j = 0; j < i; j++)
              {
//                integral_bc += cube.integrate_over_colored_interface(bc_value, j, i)/(1.0-0.0*robin_coef_p[i][n]*phi_000[j]);
//                if (phi_p[j][n] < 0)
                integral_bc += cube_refined.integrate_over_colored_interface(bc_value_refined.data(), j, i)*(1.0+.0*robin_coef_p[i][n]*phi_p[j][n]);
//                else
//                integral_bc += cube_refined.integrate_over_colored_interface(bc_value_refined.data(), j, i)/(1.0-.0*robin_coef_p[i][n]*phi_p[j][n]);
              }
            } else {
//              integral_bc = cube.integrate_over_interface(bc_value, i)/(1.0-0.0*robin_coef_p[i][n]*phi_000[i]);
//              if (phi_p[i][n] < 0)
              integral_bc = cube_refined.integrate_over_interface(bc_value_refined.data(), i)*(1.0+.0*robin_coef_p[i][n]*phi_p[i][n]);
//              else
//              integral_bc = cube_refined.integrate_over_interface(bc_value_refined.data(), i)/(1.0-.0*robin_coef_p[i][n]*phi_p[i][n]);
            }
#endif

            if((*bc_)[i].interfaceType() == NEUMANN)
              rhs_p[n] += mu_*integral_bc;
            else
              rhs_p[n] += mu_*integral_bc;
            }
          }

//          if (is_node_xmWall(p4est, ni)) rhs_p[n] += mu_*s_m00*bc_->wallValue(x_C, y_C);
//          if (is_node_xpWall(p4est, ni)) rhs_p[n] += mu_*s_p00*bc_->wallValue(x_C, y_C);
//          if (is_node_ymWall(p4est, ni)) rhs_p[n] += mu_*s_0m0*bc_->wallValue(x_C, y_C);
//          if (is_node_ypWall(p4est, ni)) rhs_p[n] += mu_*s_0p0*bc_->wallValue(x_C, y_C);
          rhs_p[n] /= w_000;
#endif
        }
      } break;
      }

    }
  }

  if (matrix_has_nullspace && fixed_value_idx_l >= 0){
    rhs_p[fixed_value_idx_l] = 0;
  }

  ierr = VecRestoreArray(add_, &add_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs_, &rhs_p); CHKERRXX(ierr);
  ierr = VecDestroy(rhs_dup); CHKERRXX(ierr);

  if (keep_scalling) {ierr = VecRestoreArray(scalling, &scalling_p); CHKERRXX(ierr);}

  for (int i = 0; i < n_phi; i++)
  {
    if (robin_coef_p[i]) {ierr = VecRestoreArray((*robin_coef_)[i], &robin_coef_p[i]);  CHKERRXX(ierr);}
                          ierr = VecRestoreArray((*phi_)[i],        &phi_p[i]);         CHKERRXX(ierr);
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_mls_t::set_phi(std::vector<Vec> *phi, std::vector<Vec> *phi_xx, std::vector<Vec> *phi_yy, std::vector<Vec> *phi_zz)
#else
void my_p4est_poisson_nodes_mls_t::set_phi(std::vector<Vec> *phi, std::vector<Vec> *phi_xx, std::vector<Vec> *phi_yy)
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
    if (phi_xx_ != NULL && is_phi_dd_owned)
    {
      for (int i = 0; i < phi_xx_->size(); i++)
        ierr = VecDestroy(phi_xx_->at(i)); CHKERRXX(ierr);
      delete phi_xx_;
    }
    phi_xx_ = new std::vector<Vec>();

    if (phi_yy_ != NULL && is_phi_dd_owned)
    {
      for (int i = 0; i < phi_yy_->size(); i++)
        ierr = VecDestroy(phi_yy_->at(i)); CHKERRXX(ierr);
      delete phi_yy_;
    }
    phi_yy_ = new std::vector<Vec>();

#ifdef P4_TO_P8
    if (phi_zz_ != NULL && is_phi_dd_owned)
    {
      for (int i = 0; i < phi_zz_->size(); i++)
        ierr = VecDestroy(phi_zz_->at(i)); CHKERRXX(ierr);
      delete phi_zz_;
    }
    phi_zz_ = new std::vector<Vec>();
#endif
    for (unsigned int i = 0; i < phi_->size(); i++)
    {
      phi_xx_->push_back(Vec()); ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx_->at(i)); CHKERRXX(ierr);
      phi_yy_->push_back(Vec()); ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy_->at(i)); CHKERRXX(ierr);
#ifdef P4_TO_P8
      phi_zz_->push_back(Vec()); ierr = VecCreateGhostNodes(p4est, nodes, &phi_zz_->at(i)); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
      node_neighbors_->second_derivatives_central(phi_->at(i), phi_xx_->at(i), phi_yy_->at(i), phi_zz_->at(i));
#else
      node_neighbors_->second_derivatives_central(phi_->at(i), phi_xx_->at(i), phi_yy_->at(i));
#endif
    }
    is_phi_dd_owned = true;
  }

  // set up effective lsf
  int n_phi = phi_->size();
  ierr = VecCreateGhostNodes(p4est, nodes, &phi_eff_); CHKERRXX(ierr);

  std::vector<double *>   phi_p(n_phi, NULL);
  double                  *phi_eff_p;

  for (int i = 0; i < n_phi; i++) { ierr = VecGetArray((*phi_)[i], &phi_p[i]);  CHKERRXX(ierr);}
                                    ierr = VecGetArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    phi_eff_p[n] = -1.;

    for (int i = 0; i < n_phi; i++)
    {
      switch ((*action_)[i])
      {
      case INTERSECTION:  phi_eff_p[n] = (phi_eff_p[n] > phi_p[i][n]) ? phi_eff_p[n] : phi_p[i][n]; break;
      case ADDITION:      phi_eff_p[n] = (phi_eff_p[n] < phi_p[i][n]) ? phi_eff_p[n] : phi_p[i][n]; break;
      case COLORATION:    /* do nothing */ break;
      }
    }
  }

  for (int i = 0; i < n_phi; i++) { ierr = VecRestoreArray((*phi_)[i], &phi_p[i]);  CHKERRXX(ierr);}
                                    ierr = VecRestoreArray(phi_eff_, &phi_eff_p); CHKERRXX(ierr);

  ierr = VecGhostUpdateBegin(phi_eff_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd  (phi_eff_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_mls_t::set_mu(Vec mue, Vec mue_xx, Vec mue_yy, Vec mue_zz)
#else
void my_p4est_poisson_nodes_mls_t::set_mu(Vec mue, Vec mue_xx, Vec mue_yy)
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

void my_p4est_poisson_nodes_mls_t::shift_to_exact_solution(Vec sol, Vec uex){
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
    if (global_node_offset[r] <= fixed_value_idx_g && fixed_value_idx_g < global_node_offset[r+1]){
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

void my_p4est_poisson_nodes_mls_t::construct_domain()
{
  // determine what BCs are present
  bool all_neumann    = true;
  bool all_dirichlet  = true;

  double eps = 1E-6*d_min*d_min;

  for (int i = 0; i < bc_->size(); i++)
  {
    all_dirichlet = all_dirichlet && (bc_->at(i).interfaceType() == DIRICHLET);
    all_neumann   = all_neumann   && (bc_->at(i).interfaceType() == ROBIN ||
                                      bc_->at(i).interfaceType() == NEUMANN);
  }

//  bool mixed_bc = !all_dirichlet && !all_neumann;

//  double eps = 1E-6*d_min*d_min;

  int n_phi = phi_->size(); // number of level set functions

  double **phi_p = new double* [n_phi];
  for (int i_phi = 0; i_phi < n_phi; i_phi++) {ierr = VecGetArray(phi_->at(i_phi), &phi_p[i_phi]); CHKERRXX(ierr);}

//  node_loc.resize(nodes->num_owned_indeps, OUT);

  if (all_dirichlet)
  {
  }
  else if (all_neumann)
  {
    double *phi_cube = new double [n_phi*P4EST_CHILDREN];
//    std::vector<double> phi_cube(n_phi*P4EST_CHILDREN, 0);

//    int n_tot = nodes->num_owned_indeps;
    cubes.reserve(nodes->num_owned_indeps);

#ifdef P4_TO_P8
    cube3_mls_t *cube;
#else
    cube2_mls_t *cube;
#endif

    for(p4est_locidx_t n = 0; n < nodes->num_owned_indeps; n++) // loop over nodes
    {
      // tree information
      p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

      double x_C  = node_x_fr_n(n, p4est, nodes);
      double y_C  = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z_C  = node_z_fr_n(n, p4est, nodes);
#endif

      // create a cube
#ifdef P4_TO_P8
      cubes.push_back(cube3_mls_t());
#else
      cubes.push_back(cube2_mls_t());
#endif
      cube = &cubes.back();

      cube->x0 = is_node_xmWall(p4est, ni) ? x_C : x_C-0.5*dx_min;
      cube->x1 = is_node_xpWall(p4est, ni) ? x_C : x_C+0.5*dx_min;
      cube->y0 = is_node_ymWall(p4est, ni) ? y_C : y_C-0.5*dy_min;
      cube->y1 = is_node_ypWall(p4est, ni) ? y_C : y_C+0.5*dy_min;
#ifdef P4_TO_P8
      cube->z0 = is_node_zmWall(p4est, ni) ? z_C : z_C-0.5*dz_min;
      cube->z1 = is_node_zpWall(p4est, ni) ? z_C : z_C+0.5*dz_min;
#endif

      // interpolate all level set functions to the cube
      for (int i_phi = 0; i_phi < n_phi; i_phi++){
#ifdef P4_TO_P8
        phi_interp.set_input(phi_->at(i_phi), phi_xx_->at(i_phi), phi_yy_->at(i_phi), phi_zz_->at(i_phi), linear);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = phi_interp(cube->x0, cube->y0, cube->z0);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmp] = phi_interp(cube->x0, cube->y0, cube->z1);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = phi_interp(cube->x0, cube->y1, cube->z0);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpp] = phi_interp(cube->x0, cube->y1, cube->z1);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = phi_interp(cube->x1, cube->y0, cube->z0);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmp] = phi_interp(cube->x1, cube->y0, cube->z1);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = phi_interp(cube->x1, cube->y1, cube->z0);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppp] = phi_interp(cube->x1, cube->y1, cube->z1);
#else
//        phi_interp.set_input(phi_->at(i_phi), phi_xx_->at(i_phi), phi_yy_->at(i_phi), linear);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = phi_interp(cube->x0, cube->y0);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = phi_interp(cube->x0, cube->y1);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = phi_interp(cube->x1, cube->y0);
//        phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = phi_interp(cube->x1, cube->y1);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mmm] = (*phi_cf_->at(i_phi))(cube->x0, cube->y0);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_mpm] = (*phi_cf_->at(i_phi))(cube->x0, cube->y1);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_pmm] = (*phi_cf_->at(i_phi))(cube->x1, cube->y0);
        phi_cube[i_phi*P4EST_CHILDREN + dir::v_ppm] = (*phi_cf_->at(i_phi))(cube->x1, cube->y1);
#endif
      }

      // construct geometry inside the cube
      cube->construct_domain(phi_cube, *action_, *color_);

      // temporary: validating function measure_in_dir

//      if (cube->measure_of_domain() < eps*eps)
//      {
////        std::cout << "yup!\n";
//        cube->loc = OUT;
//      }

#ifdef P4_TO_P8
      if (cube->loc == FCE)
      {
        double *phi_face = new double [4*n_phi];
        int v0, v1, v2, v3;
        cube2_mls_t cube_aux;

        // -x dir
        v0 = 0; v1 = 2; v2 = 4; v3 = 6;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->y0; cube_aux.x1 = cube->y1;
        cube_aux.y0 = cube->z0; cube_aux.y1 = cube->z1;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(0));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }

        // +x dir
        v0 = 1; v1 = 3; v2 = 5; v3 = 7;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->y0; cube_aux.x1 = cube->y1;
        cube_aux.y0 = cube->z0; cube_aux.y1 = cube->z1;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(1));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }

        // -y dir
        v0 = 0; v1 = 1; v2 = 4; v3 = 5;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->x0; cube_aux.x1 = cube->x1;
        cube_aux.y0 = cube->z0; cube_aux.y1 = cube->z1;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(2));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }

        // +y dir
        v0 = 2; v1 = 3; v2 = 6; v3 = 7;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->x0; cube_aux.x1 = cube->x1;
        cube_aux.y0 = cube->z0; cube_aux.y1 = cube->z1;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(3));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }

        // -z dir
        v0 = 0; v1 = 1; v2 = 2; v3 = 3;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->x0; cube_aux.x1 = cube->x1;
        cube_aux.y0 = cube->y0; cube_aux.y1 = cube->y1;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(4));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }

        // +z dir
        v0 = 4; v1 = 5; v2 = 6; v3 = 7;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->x0; cube_aux.x1 = cube->x1;
        cube_aux.y0 = cube->y0; cube_aux.y1 = cube->y1;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(5));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }
      }
#else
      if (cube->loc == FCE)
      {
        double *phi_face = new double [4*n_phi];
        int v0, v1, v2, v3;
        cube2_mls_t cube_aux;

        // -x dir
        v0 = 0; v1 = 2; v2 = 0; v3 = 2;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->y0; cube_aux.x1 = cube->y1;
        cube_aux.y0 = 0.0; cube_aux.y1 = 1.0;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(0));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }

        // +x dir
        v0 = 1; v1 = 3; v2 = 1; v3 = 3;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->y0; cube_aux.x1 = cube->y1;
        cube_aux.y0 = 0.0; cube_aux.y1 = 1.0;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(1));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }

        // -y dir
        v0 = 0; v1 = 1; v2 = 0; v3 = 1;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->x0; cube_aux.x1 = cube->x1;
        cube_aux.y0 = 0.0; cube_aux.y1 = 1.0;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(2));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }

        // +y dir
        v0 = 2; v1 = 3; v2 = 2; v3 = 3;
        for (int i_phi = 0; i_phi < n_phi; i_phi++){
          phi_face[i_phi*4 + 0] = phi_cube[i_phi*P4EST_CHILDREN + v0];
          phi_face[i_phi*4 + 1] = phi_cube[i_phi*P4EST_CHILDREN + v1];
          phi_face[i_phi*4 + 2] = phi_cube[i_phi*P4EST_CHILDREN + v2];
          phi_face[i_phi*4 + 3] = phi_cube[i_phi*P4EST_CHILDREN + v3];
        }
        cube_aux.x0 = cube->x0; cube_aux.x1 = cube->x1;
        cube_aux.y0 = 0.0; cube_aux.y1 = 1.0;
        cube_aux.construct_domain(phi_face, *action_, *color_);
        if (cube_aux.loc == FCE)
        {
          double dif = fabs(cube_aux.measure_of_domain() - cube->measure_in_dir(3));
          if (dif > 1.e-16) throw std::domain_error("[CASL_ERROR]: error.");
        }
      }
#endif

      switch (cube->loc){
      case INS: node_loc[n] = NODE_INS;  break;
      case FCE: node_loc[n] = NODE_NMN;  break;
      case OUT: node_loc[n] = NODE_OUT;  break;
      }
    }

    delete[] phi_cube;

  } else {
    /*

    double *phi_cube = new double [n_phi*P4EST_CHILDREN];

    for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
    {
      // tree information
      p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

      //---------------------------------------------------------------------
      // Information at neighboring nodes
      //---------------------------------------------------------------------
      double x_C  = node_x_fr_n(n, p4est, nodes);
      double y_C  = node_y_fr_n(n, p4est, nodes);
#ifdef P4_TO_P8
      double z_C  = node_z_fr_n(n, p4est, nodes);
#endif

      const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

      double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
#ifdef P4_TO_P8
      double phi_00m, phi_00p;
#endif

#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
#endif

      // TODO: This needs optimization
      // create a cube
#ifdef P4_TO_P8
      MLS_Cube3 cube;
#else
      MLS_Cube2 cube;
#endif
      cube.x0 = is_node_xmWall(p4est, ni) ? x_C : x_C-0.5*dx_min;
      cube.x1 = is_node_xpWall(p4est, ni) ? x_C : x_C+0.5*dx_min;
      cube.y0 = is_node_ymWall(p4est, ni) ? y_C : y_C-0.5*dy_min;
      cube.y1 = is_node_ypWall(p4est, ni) ? y_C : y_C+0.5*dy_min;
#ifdef P4_TO_P8
      cube.z0 = is_node_zmWall(p4est, ni) ? z_C : z_C-0.5*dz_min;
      cube.z1 = is_node_zpWall(p4est, ni) ? z_C : z_C+0.5*dz_min;
#endif

      // interpolate all level set functions to the cube
      for (int i_phi = 0; i_phi < n_phi; i_phi++){
#ifdef P4_TO_P8
        phi_cube[i_phi*dir::v_mmm] = phi_interp(cube.x0, cube.y0, cube.z0);
        phi_cube[i_phi*dir::v_mmp] = phi_interp(cube.x0, cube.y0, cube.z1);
        phi_cube[i_phi*dir::v_mpm] = phi_interp(cube.x0, cube.y1, cube.z0);
        phi_cube[i_phi*dir::v_mpp] = phi_interp(cube.x0, cube.y1, cube.z1);
        phi_cube[i_phi*dir::v_pmm] = phi_interp(cube.x1, cube.y0, cube.z0);
        phi_cube[i_phi*dir::v_pmp] = phi_interp(cube.x1, cube.y0, cube.z1);
        phi_cube[i_phi*dir::v_ppm] = phi_interp(cube.x1, cube.y1, cube.z0);
        phi_cube[i_phi*dir::v_ppp] = phi_interp(cube.x1, cube.y1, cube.z1);
#else
        phi_cube[i_phi*dir::v_mmm] = phi_interp(cube.x0, cube.y0);
        phi_cube[i_phi*dir::v_mpm] = phi_interp(cube.x0, cube.y1);
        phi_cube[i_phi*dir::v_pmm] = phi_interp(cube.x1, cube.y0);
        phi_cube[i_phi*dir::v_ppm] = phi_interp(cube.x1, cube.y1);
#endif
      }

      //
      cube.construct_domain(phi_cube, action_, color_);

      if (cube.location == interface){

        all_neumann   = true;
        all_dirichlet = true;

        // check which boundaries are present within the cube
        for (int i_phi = 0; i_phi < n_phi; i_phi++){

          double int_measure = cube.interface_measure(i_phi);

          if (int_measure > EPS){
            all_dirichlet = all_dirichlet || bc_->at(color_->at(i_phi)) == DIRICHLET;
            all_neumann   = all_neumann   || bc_->at(color_->at(i_phi)) == NEUMANN ||
                            bc_->at(color_->at(i_phi)) == ROBIN;

            if (bc_->at(i_phi) == DIRICHLET){ // remove dirichlet boundaries
              if (phi_p[i_phi][n] < 0) for (int i = 0; i < P4EST_CHILDREN; i++) phi_cube[i_phi*i] = -1.0;
              else                     for (int i = 0; i < P4EST_CHILDREN; i++) phi_cube[i_phi*i] =  1.0;
            }
          }

          if (int_measure < EPS){ // remove not present boundaries
            for (int i = 0; i < P4EST_CHILDREN; i++)
              switch (action_->at(i_phi)){
              case intersection:  phi_cube[i_phi*i] = -1.0; break;
              case addition:      phi_cube[i_phi*i] =  1.0; break;
              case coloration:    phi_cube[i_phi*i] =  1.0; break;
              }
          }
          // FIXME: coloration doesn't work yet
        }

        // if mixed bc then reconstruct domain after removing
        if (all_neumann && all_dirichlet) cube.construct_domain(phi_cube, action_, color_);

        if (cube.location == interface){
          if (all_dirichlet)    node_loc[n] = DIRICHLET;
          else if (all_neumann) node_loc[n] = NEUMANN;
          else                  node_loc[n] = MIXED_IN;
        } else if (cube.location == inside)
      }

      if (cube.location == inside)        node_loc[n] = INSIDE;
      else if (cube.location == outside)  node_loc[n] = OUTSIDE;



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
        bool is_interface_m00 = (bc_->interfaceType() == DIRICHLET && phi_m00*phi_000 <= 0.);
        bool is_interface_p00 = (bc_->interfaceType() == DIRICHLET && phi_p00*phi_000 <= 0.);
        bool is_interface_0m0 = (bc_->interfaceType() == DIRICHLET && phi_0m0*phi_000 <= 0.);
        bool is_interface_0p0 = (bc_->interfaceType() == DIRICHLET && phi_0p0*phi_000 <= 0.);
#ifdef P4_TO_P8
        bool is_interface_00m = (bc_->interfaceType() == DIRICHLET && phi_00m*phi_000 <= 0.);
        bool is_interface_00p = (bc_->interfaceType() == DIRICHLET && phi_00p*phi_000 <= 0.);
#endif
      }
    }
    */
  }


  for (int i_phi = 0; i_phi < n_phi; i_phi++) {ierr = VecRestoreArray(phi_->at(i_phi), &phi_p[i_phi]); CHKERRXX(ierr);}

  delete[] phi_p;

}
