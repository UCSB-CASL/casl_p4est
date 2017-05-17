#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
#else
#include "my_p4est_poisson_nodes.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/cube2.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>

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

//#ifdef P4_TO_P8
//#define 3_IN_D 27
//#else
//#define 3_IN_D 9
//#endif


my_p4est_poisson_nodes_t::my_p4est_poisson_nodes_t(const my_p4est_node_neighbors_t *node_neighbors)
  : node_neighbors_(node_neighbors),
    p4est(node_neighbors->p4est), nodes(node_neighbors->nodes), ghost(node_neighbors->ghost), myb_(node_neighbors->myb),
    phi_interp(node_neighbors), //robin_coef_interp(node_neighbors),
    neumann_wall_first_order(false),
    mu_(1.), diag_add_(0.),
    is_matrix_computed(false), matrix_has_nullspace(false),
    bc_(NULL), A(NULL),
    is_phi_dd_owned(false), is_mue_dd_owned(false),
    rhs_(NULL), phi_(NULL), add_(NULL), mue_(NULL),
    phi_xx_(NULL), phi_yy_(NULL), mue_xx_(NULL), mue_yy_(NULL),
    use_refined_cube(true), variable_mu(false), use_pointwise_dirichlet(false),
    mask(NULL),
    keep_scalling(false),
    new_pc(true)
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
  // NOTE: Assuming all trees are of the same size. Must be generalized if different trees have
  // different sizes
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

  dxyz_m[0] = dx_min;
  dxyz_m[1] = dy_min;
#ifdef P4_TO_P8
  dxyz_m[2] = dz_min;
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

  scalling.resize(nodes->num_owned_indeps, 1);
//  pointwise_bc.resize(nodes->num_owned_indeps);
}

my_p4est_poisson_nodes_t::~my_p4est_poisson_nodes_t()
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

  if (mask != NULL) { ierr = VecDestroy(mask); CHKERRXX(ierr); }
}

void my_p4est_poisson_nodes_t::preallocate_matrix()
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

    if (!bc_->interfaceType() == NOINTERFACE)
      if (phi_p[n] > 2*diag_min)
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

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
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
  bool local_phi = false;
  if(phi_ == NULL)
  {
    local_phi = true;
    ierr = VecDuplicate(solution, &phi_); CHKERRXX(ierr);

    Vec tmp;
    ierr = VecGhostGetLocalForm(phi_, &tmp); CHKERRXX(ierr);
    ierr = VecSet(tmp, -1.); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi_, &tmp); CHKERRXX(ierr);
//    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi_); CHKERRXX(ierr);
    set_phi(phi_);
  }

  bool local_rhs = false;
  if (rhs_ == NULL)
  {
    ierr = VecDuplicate(solution, &rhs_); CHKERRXX(ierr);
    Vec rhs_local;
    VecGhostGetLocalForm(rhs_, &rhs_local);
    VecSet(rhs_local, 0);
    VecGhostRestoreLocalForm(rhs_, &rhs_local);
    local_rhs = true;
  }

  // set ksp type
  ierr = KSPSetType(ksp, ksp_type); CHKERRXX(ierr);  
  if (use_nonzero_initial_guess)
    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); CHKERRXX(ierr);

  /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
  if (!is_matrix_computed)
  {
    matrix_has_nullspace = true;
    setup_negative_variable_coeff_laplace_matrix();
    is_matrix_computed = true;
    new_pc = true;
  }

  if (new_pc)
  {
    new_pc = false;
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

    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
    if (matrix_has_nullspace){
      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
    }
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // setup rhs
  setup_negative_variable_coeff_laplace_rhsvec();

  // Solve the system
  ierr = KSPSetTolerances(ksp, 1e-16, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRXX(ierr);

  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_KSPSolve, ksp, rhs_, solution, 0); CHKERRXX(ierr);  
  MatNullSpace A_null;
  if (matrix_has_nullspace) {
    ierr = MatNullSpaceCreate(p4est->mpicomm, PETSC_TRUE, 0, NULL, &A_null); CHKERRXX(ierr);
    ierr = MatSetNullSpace(A, A_null);

    // For purely neumann problems GMRES is more robust
    ierr = KSPSetType(ksp, KSPGMRES); CHKERRXX(ierr);
  }

  ierr = KSPSolve(ksp, rhs_, solution); CHKERRXX(ierr);
  if (matrix_has_nullspace) {
    ierr = MatNullSpaceDestroy(A_null);
  }
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
  if(local_rhs)
  {
    ierr = VecDestroy(rhs_); CHKERRXX(ierr);
  }
  if(local_phi)
  {
    ierr = VecDestroy(phi_); CHKERRXX(ierr);
    phi_ = NULL;

    ierr = VecDestroy(phi_xx_); CHKERRXX(ierr);
    phi_xx_ = NULL;

    ierr = VecDestroy(phi_yy_); CHKERRXX(ierr);
    phi_yy_ = NULL;

#ifdef P4_TO_P8
    ierr = VecDestroy(phi_zz_); CHKERRXX(ierr);
    phi_zz_ = NULL;
#endif
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_solve, A, rhs_, ksp, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_t::setup_negative_variable_coeff_laplace_matrix()
{
  if (use_pointwise_dirichlet)
  {
    pointwise_bc.clear();
    pointwise_bc.resize(nodes->num_owned_indeps);
  }

  preallocate_matrix();

  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

#ifdef P4_TO_P8
  double eps = 1E-6*d_min*d_min*d_min;
#else
  double eps = 1E-6*d_min*d_min;
#endif

  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p;
  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif

  double *mue_p=NULL, *mue_xx_p=NULL, *mue_yy_p=NULL, *mue_zz_p=NULL;
  if (variable_mu)
  {
    ierr = VecGetArray(mue_,    &mue_p   ); CHKERRXX(ierr);
    ierr = VecGetArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
    ierr = VecGetArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  }

  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);

  double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
  double mue_000, mue_p00, mue_m00, mue_0m0, mue_0p0;

#ifdef P4_TO_P8
  double phi_00m, phi_00p;
  double mue_00m, mue_00p;
#endif

  if (!variable_mu) {
    mue_000 = mu_; mue_p00 = mu_; mue_m00 = mu_; mue_0m0 = mu_; mue_0p0 = mu_;
#ifdef P4_TO_P8
    mue_00m = mu_; mue_00p = mu_;
#endif
  }

  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);
  phi_interp_local.set_input(phi_p, phi_xx_p, phi_yy_p, quadratic);

  /* Daniil: while working on solidification of alloys I came to realize
   * that due to very irregular structure of solidification fronts (where
   * there constantly exist nodes, which belong to one phase but the vertices
   * of their dual cells belong to the other phase. In such cases simple dual
   * cells consisting of 2 (6 in 3D) simplices don't capture such complex
   * topologies. To aleviate this issue I added an option to use a more detailed
   * dual cells consisting of 8 (48 in 3D) simplices. Clearly, it might increase
   * the computational cost quite significantly, but oh well...
   */

  // data for refined cells
  std::vector<double> phi_fv(pow(2*cube_refinement+1,P4EST_DIM),-1);
  double fv_size_x = 0, fv_nx; std::vector<double> fv_x(2*cube_refinement+1, 0);
  double fv_size_y = 0, fv_ny; std::vector<double> fv_y(2*cube_refinement+1, 0);
#ifdef P4_TO_P8
  double fv_size_z = 0, fv_nz; std::vector<double> fv_z(2*cube_refinement+1, 0);
#endif

  double fv_xmin, fv_xmax;
  double fv_ymin, fv_ymax;
#ifdef P4_TO_P8
  double fv_zmin, fv_zmax;
#endif

  double xyz_C[P4EST_DIM];

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    node_xyz_fr_n(n, p4est, nodes, xyz_C);
    double x_C  = xyz_C[0];
    double y_C  = xyz_C[1];
#ifdef P4_TO_P8
    double z_C  = xyz_C[2];
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

    double w_m00_mm=0, w_m00_pm=0;
    double w_p00_mm=0, w_p00_pm=0;
    double w_0m0_mm=0, w_0m0_pm=0;
    double w_0p0_mm=0, w_0p0_pm=0;
#ifdef P4_TO_P8
    double d_m00_0m=qnnn.d_m00_0m; double d_m00_0p=qnnn.d_m00_0p;
    double d_p00_0m=qnnn.d_p00_0m; double d_p00_0p=qnnn.d_p00_0p;
    double d_0m0_0m=qnnn.d_0m0_0m; double d_0m0_0p=qnnn.d_0m0_0p;
    double d_0p0_0m=qnnn.d_0p0_0m; double d_0p0_0p=qnnn.d_0p0_0p;

    double d_00m_m0=qnnn.d_00m_m0; double d_00m_p0=qnnn.d_00m_p0;
    double d_00p_m0=qnnn.d_00p_m0; double d_00p_p0=qnnn.d_00p_p0;
    double d_00m_0m=qnnn.d_00m_0m; double d_00m_0p=qnnn.d_00m_0p;
    double d_00p_0m=qnnn.d_00p_0m; double d_00p_0p=qnnn.d_00p_0p;

    double w_m00_mp=0, w_m00_pp=0;
    double w_p00_mp=0, w_p00_pp=0;
    double w_0m0_mp=0, w_0m0_pp=0;
    double w_0p0_mp=0, w_0p0_pp=0;

    double w_00m_mm=0, w_00m_pm=0;
    double w_00p_mm=0, w_00p_pm=0;
    double w_00m_mp=0, w_00m_pp=0;
    double w_00p_mp=0, w_00p_pp=0;
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
      if(bc_->wallType(xyz_C) == DIRICHLET)
      {
        ierr = MatSetValue(A, node_000_g, node_000_g, bc_strength, ADD_VALUES); CHKERRXX(ierr);
        if (phi_p[n]<0. || bc_->interfaceType() == NOINTERFACE) matrix_has_nullspace = false;
        continue;
      }

      // In case if you want first order neumann at walls. Why is it still a thing anyway?
      if(bc_->wallType(xyz_C) == NEUMANN && neumann_wall_first_order)
      {
        if (is_node_xpWall(p4est, ni)){
#ifdef P4_TO_P8
          p4est_locidx_t n_m00 = d_m00_0m == 0 ? ( d_m00_m0==0 ? node_m00_mm : node_m00_pm )
                                               : ( d_m00_m0==0 ? node_m00_mp : node_m00_pp );
#else
          p4est_locidx_t n_m00 = d_m00_m0 == 0 ? node_m00_mm : node_m00_pm;
#endif
          PetscInt node_m00_g  = petsc_gloidx[n_m00];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);

          if (phi_p[n] < diag_min || bc_->interfaceType() == NOINTERFACE)
            ierr = MatSetValue(A, node_000_g, node_m00_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);

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

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min || bc_->interfaceType() == NOINTERFACE)
            ierr = MatSetValue(A, node_000_g, node_p00_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);

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

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min || bc_->interfaceType() == NOINTERFACE)
            ierr = MatSetValue(A, node_000_g, node_0m0_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);

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

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min || bc_->interfaceType() == NOINTERFACE)
            ierr = MatSetValue(A, node_000_g, node_0p0_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);

          continue;
        }
#ifdef P4_TO_P8
        if (is_node_zpWall(p4est, ni)){
          p4est_locidx_t n_00m = d_00m_0m == 0 ? ( d_00m_m0 == 0 ? node_00m_mm : node_00m_pm )
                                               : ( d_00m_m0 == 0 ? node_00m_mp : node_00m_pp );
          PetscInt node_00m_g  = petsc_gloidx[n_00m];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min || bc_->interfaceType() == NOINTERFACE)
            ierr = MatSetValue(A, node_000_g, node_00m_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);

          continue;
        }

        if (is_node_zmWall(p4est, ni)){
          p4est_locidx_t n_00p = d_00p_0m == 0 ? ( d_00p_m0 == 0 ? node_00p_mm : node_00p_pm )
                                               : ( d_00p_m0 == 0 ? node_00p_mp : node_00p_pp );
          PetscInt node_00p_g  = petsc_gloidx[n_00p];

          ierr = MatSetValue(A, node_000_g, node_000_g,  bc_strength, ADD_VALUES); CHKERRXX(ierr);
          if (phi_p[n] < diag_min || bc_->interfaceType() == NOINTERFACE)
            ierr = MatSetValue(A, node_000_g, node_00p_g, -bc_strength, ADD_VALUES); CHKERRXX(ierr);

          continue;
        }
#endif
      }

    }

    {
#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
      if (variable_mu)
        qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
      if (variable_mu)
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

      //---------------------------------------------------------------------
      // check if finite volume is crossed
      //---------------------------------------------------------------------
      bool is_one_positive = false;
      bool is_one_negative = false;

      if (fabs(phi_000) < 2.*diag_min && (bc_->interfaceType() == ROBIN || bc_->interfaceType() == NEUMANN))
//      if (fabs(phi_000) < 2.*diag_min)
      {
        phi_interp_local.initialize(n);
        // determine dimensions of cube
        fv_size_x = 0;
        fv_size_y = 0;
#ifdef P4_TO_P8
        fv_size_z = 0;
#endif
        if(!is_node_xmWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmin = x_C-0.5*dx_min;} else {fv_xmin = x_C;}
        if(!is_node_xpWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmax = x_C+0.5*dx_min;} else {fv_xmax = x_C;}

        if(!is_node_ymWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymin = y_C-0.5*dy_min;} else {fv_ymin = y_C;}
        if(!is_node_ypWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymax = y_C+0.5*dy_min;} else {fv_ymax = y_C;}
#ifdef P4_TO_P8
        if(!is_node_zmWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmin = z_C-0.5*dz_min;} else {fv_zmin = z_C;}
        if(!is_node_zpWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmax = z_C+0.5*dz_min;} else {fv_zmax = z_C;}
#endif
        if (!use_refined_cube) {
          fv_size_x = 1;
          fv_size_y = 1;
#ifdef P4_TO_P8
          fv_size_z = 1;
#endif
        }

        fv_nx = fv_size_x+1;
        fv_ny = fv_size_y+1;
#ifdef P4_TO_P8
        fv_nz = fv_size_z+1;
#endif

        // get coordinates of cube nodes
        double fv_dx = (fv_xmax-fv_xmin)/ (double)(fv_size_x);
        fv_x[0] = fv_xmin;
        for (short i = 1; i < fv_nx; ++i)
          fv_x[i] = fv_x[i-1] + fv_dx;

        double fv_dy = (fv_ymax-fv_ymin)/ (double)(fv_size_y);
        fv_y[0] = fv_ymin;
        for (short i = 1; i < fv_ny; ++i)
          fv_y[i] = fv_y[i-1] + fv_dy;
#ifdef P4_TO_P8
        double fv_dx = (fv_zmax-fv_zmin)/ (double)(fv_size_z);
        fv_z[0] = fv_zmin;
        for (short i = 1; i < fv_nz; ++i)
          fv_z[i] = fv_z[i-1] + fv_dz;
#endif

        // sample level-set function at cube nodes and check if crossed
#ifdef P4_TO_P8
        for (short k = 0; k < fv_nz; ++k)
#endif
          for (short j = 0; j < fv_ny; ++j)
            for (short i = 0; i < fv_nx; ++i)
            {
#ifdef P4_TO_P8
              int idx = k*fv_nx*fv_ny + j*fv_nx + i;
              phi_fv[idx] = phi_interp(fv_x[i], fv_y[j], fv_z[k]);
#else
              int idx = j*fv_nx + i;
//              phi_fv[idx] = phi_interp(fv_x[i], fv_y[j]);
              phi_fv[idx] = phi_interp_local.interpolate(fv_x[i], fv_y[j]);
#endif
              is_one_positive = is_one_positive || phi_fv[idx] > 0;
              is_one_negative = is_one_negative || phi_fv[idx] < 0;
            }
      }

      bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;

      // far away from the interface
      if(phi_000>0. &&  (!is_ngbd_crossed_neumann || bc_->interfaceType() == DIRICHLET ) && (bc_->interfaceType() != NOINTERFACE)){
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

          if (variable_mu) {
            double mxx_000 = mue_xx_p[n];
            double mxx_m00 = qnnn.f_m00_linear(mue_xx_p);
            mue_m00 = mue_000*(1-theta_m00/d_m00) + mue_m00*theta_m00/d_m00 + 0.5*theta_m00*(theta_m00-d_m00)*MINMOD(mxx_m00,mxx_000);
          }

          d_m00 = theta_m00;

          if (use_pointwise_dirichlet)
            pointwise_bc[n].push_back(interface_point_t(0, theta_m00));
        }
        if( is_interface_p00){
          double phixx_p00 = qnnn.f_p00_linear(phi_xx_p);
          double theta_p00 = interface_Location_With_Second_Order_Derivative(0., d_p00, phi_000, phi_p00, phixx_C, phixx_p00);
          if (theta_p00<eps) theta_p00 = eps; if (theta_p00>d_p00) theta_p00 = d_p00;
          d_p00_m0 = d_p00_p0 = 0;
#ifdef P4_TO_P8
          d_p00_0m = d_p00_0p = 0;
#endif

          if (variable_mu) {
            double mxx_000 = mue_xx_p[n];
            double mxx_p00 = qnnn.f_p00_linear(mue_xx_p);
            mue_p00 = mue_000*(1-theta_p00/d_p00) + mue_p00*theta_p00/d_p00 + 0.5*theta_p00*(theta_p00-d_p00)*MINMOD(mxx_p00,mxx_000);
          }

          d_p00 = theta_p00;

          if (use_pointwise_dirichlet)
            pointwise_bc[n].push_back(interface_point_t(1, theta_p00));
        }
        if( is_interface_0m0){
          double phiyy_0m0 = qnnn.f_0m0_linear(phi_yy_p);
          double theta_0m0 = interface_Location_With_Second_Order_Derivative(0., d_0m0, phi_000, phi_0m0, phiyy_C, phiyy_0m0);
          if (theta_0m0<eps) theta_0m0 = eps; if (theta_0m0>d_0m0) theta_0m0 = d_0m0;
          d_0m0_m0 = d_0m0_p0 = 0;
#ifdef P4_TO_P8
          d_0m0_0m = d_0m0_0p = 0;
#endif

          if (variable_mu) {
            double myy_000 = mue_yy_p[n];
            double myy_0m0 = qnnn.f_0m0_linear(mue_yy_p);
            mue_0m0 = mue_000*(1-theta_0m0/d_0m0) + mue_0m0*theta_0m0/d_0m0 + 0.5*theta_0m0*(theta_0m0-d_0m0)*MINMOD(myy_0m0,myy_000);
          }

          d_0m0 = theta_0m0;

          if (use_pointwise_dirichlet)
            pointwise_bc[n].push_back(interface_point_t(2, theta_0m0));
        }
        if( is_interface_0p0){
          double phiyy_0p0 = qnnn.f_0p0_linear(phi_yy_p);
          double theta_0p0 = interface_Location_With_Second_Order_Derivative(0., d_0p0, phi_000, phi_0p0, phiyy_C, phiyy_0p0);
          if (theta_0p0<eps) theta_0p0 = eps; if (theta_0p0>d_0p0) theta_0p0 = d_0p0;
          d_0p0_m0 = d_0p0_p0 = 0;
#ifdef P4_TO_P8
          d_0p0_0m = d_0p0_0p = 0;
#endif

          if (variable_mu) {
            double myy_000 = mue_yy_p[n];
            double myy_0p0 = qnnn.f_0p0_linear(mue_yy_p);
            mue_0p0 = mue_000*(1-theta_0p0/d_0p0) + mue_0p0*theta_0p0/d_0p0 + 0.5*theta_0p0*(theta_0p0-d_0p0)*MINMOD(myy_0p0,myy_000);
          }

          d_0p0 = theta_0p0;

          if (use_pointwise_dirichlet)
            pointwise_bc[n].push_back(interface_point_t(3, theta_0p0));
        }
#ifdef P4_TO_P8
        if( is_interface_00m){
          double phizz_00m = qnnn.f_00m_linear(phi_zz_p);
          double theta_00m = interface_Location_With_Second_Order_Derivative(0., d_00m, phi_000, phi_00m, phizz_C, phizz_00m);
          if (theta_00m<eps) theta_00m = eps; if (theta_00m>d_00m) theta_00m = d_00m;
          d_00m_m0 = d_00m_p0 = d_00m_0m = d_00m_0p = 0;

          if (variable_mu) {
            double mzz_000 = mue_zz_p[n];
            double mzz_00m = qnnn.f_00m_linear(mue_zz_p);
            mue_00m = mue_000*(1-theta_00m/d_00m) + mue_00m*theta_00m/d_00m + 0.5*theta_00m*(theta_00m-d_00m)*MINMOD(mzz_00m,mzz_000);
          }

          d_00m = theta_00m;

          if (use_pointwise_dirichlet)
            pointwise_bc[n].push_back(interface_point_t(4, theta_00m));
        }
        if( is_interface_00p){
          double phizz_00p = qnnn.f_00p_linear(phi_zz_p);
          double theta_00p = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000, phi_00p, phizz_C, phizz_00p);
          if (theta_00p<eps) theta_00p = eps; if (theta_00p>d_00p) theta_00p = d_00p;
          d_00p_m0 = d_00p_p0 = d_00p_0m = d_00p_0p = 0;

          if (variable_mu) {
            double mzz_000 = mue_zz_p[n];
            double mzz_00p = qnnn.f_00p_linear(mue_zz_p);
            mue_00p = mue_000*(1-theta_00p/d_00p) + mue_00p*theta_00p/d_00p + 0.5*theta_00p*(theta_00p-d_00p)*MINMOD(mzz_00p,mzz_000);
          }

          d_00p = theta_00p;

          if (use_pointwise_dirichlet)
            pointwise_bc[n].push_back(interface_point_t(5, theta_00p));
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
        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0, w_00m=0, w_00p=0;

        // if node is at wall, what's below will apply Neumann BC
        if(is_node_xmWall(p4est, ni))      w_p00 += -1.0/(d_p00*d_p00);
        else if(is_node_xpWall(p4est, ni)) w_m00 += -1.0/(d_m00*d_m00);
        else                               w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

        if(is_node_xpWall(p4est, ni))      w_m00 += -1.0/(d_m00*d_m00);
        else if(is_node_xmWall(p4est, ni)) w_p00 += -1.0/(d_p00*d_p00);
        else                               w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

        if(is_node_ymWall(p4est, ni))      w_0p0 += -1.0/(d_0p0*d_0p0);
        else if(is_node_ypWall(p4est, ni)) w_0m0 += -1.0/(d_0m0*d_0m0);
        else                               w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

        if(is_node_ypWall(p4est, ni))      w_0m0 += -1.0/(d_0m0*d_0m0);
        else if(is_node_ymWall(p4est, ni)) w_0p0 += -1.0/(d_0p0*d_0p0);
        else                               w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

        if(is_node_zmWall(p4est, ni))      w_00p += -1.0/(d_00p*d_00p);
        else if(is_node_zpWall(p4est, ni)) w_00m += -1.0/(d_00m*d_00m);
        else                               w_00m += -2.0*wk/d_00m/(d_00m+d_00p);

        if(is_node_zpWall(p4est, ni))      w_00m += -1.0/(d_00m*d_00m);
        else if(is_node_zmWall(p4est, ni)) w_00p += -1.0/(d_00p*d_00p);
        else                               w_00p += -2.0*wk/d_00p/(d_00m+d_00p);

        if (variable_mu)
        {
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

        } else {

          if(!is_interface_m00) {
            w_m00_mm = mu_*w_m00*d_m00_p0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_mp = mu_*w_m00*d_m00_p0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_pm = mu_*w_m00*d_m00_m0*d_m00_0p/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00_pp = mu_*w_m00*d_m00_m0*d_m00_0m/(d_m00_m0+d_m00_p0)/(d_m00_0m+d_m00_0p);
            w_m00 = w_m00_mm + w_m00_mp + w_m00_pm + w_m00_pp;
          } else {
            w_m00 *= mu_;
          }

          if(!is_interface_p00) {
            w_p00_mm = mu_*w_p00*d_p00_p0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_mp = mu_*w_p00*d_p00_p0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_pm = mu_*w_p00*d_p00_m0*d_p00_0p/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00_pp = mu_*w_p00*d_p00_m0*d_p00_0m/(d_p00_m0+d_p00_p0)/(d_p00_0m+d_p00_0p);
            w_p00 = w_p00_mm + w_p00_mp + w_p00_pm + w_p00_pp;
          } else {
            w_p00 *= mu_;
          }

          if(!is_interface_0m0) {
            w_0m0_mm = mu_*w_0m0*d_0m0_p0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_mp = mu_*w_0m0*d_0m0_p0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_pm = mu_*w_0m0*d_0m0_m0*d_0m0_0p/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0_pp = mu_*w_0m0*d_0m0_m0*d_0m0_0m/(d_0m0_m0+d_0m0_p0)/(d_0m0_0m+d_0m0_0p);
            w_0m0 = w_0m0_mm + w_0m0_mp + w_0m0_pm + w_0m0_pp;
          } else {
            w_0m0 *= mu_;
          }

          if(!is_interface_0p0) {
            w_0p0_mm = mu_*w_0p0*d_0p0_p0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_mp = mu_*w_0p0*d_0p0_p0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_pm = mu_*w_0p0*d_0p0_m0*d_0p0_0p/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0_pp = mu_*w_0p0*d_0p0_m0*d_0p0_0m/(d_0p0_m0+d_0p0_p0)/(d_0p0_0m+d_0p0_0p);
            w_0p0 = w_0p0_mm + w_0p0_mp + w_0p0_pm + w_0p0_pp;
          } else {
            w_0p0 *= mu_;
          }

          if(!is_interface_00m) {
            w_00m_mm = mu_*w_00m*d_00m_p0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_mp = mu_*w_00m*d_00m_p0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_pm = mu_*w_00m*d_00m_m0*d_00m_0p/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m_pp = mu_*w_00m*d_00m_m0*d_00m_0m/(d_00m_m0+d_00m_p0)/(d_00m_0m+d_00m_0p);
            w_00m = w_00m_mm + w_00m_mp + w_00m_pm + w_00m_pp;
          } else {
            w_00m *= mu_;
          }

          if(!is_interface_00p) {
            w_00p_mm = mu_*w_00p*d_00p_p0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p_mp = mu_*w_00p*d_00p_p0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p_pm = mu_*w_00p*d_00p_m0*d_00p_0p/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p_pp = mu_*w_00p*d_00p_m0*d_00p_0m/(d_00p_m0+d_00p_p0)/(d_00p_0m+d_00p_0p);
            w_00p = w_00p_mm + w_00p_mp + w_00p_pm + w_00p_pp;
          } else {
            w_00p *= mu_;
          }

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
        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;

        // if node is at wall, what's below will apply Neumann BC
        if(is_node_xmWall(p4est, ni))      w_p00 += -1.0/(d_p00*d_p00);
        else if(is_node_xpWall(p4est, ni)) w_m00 += -1.0/(d_m00*d_m00);
        else                               w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

        if(is_node_xpWall(p4est, ni))      w_m00 += -1.0/(d_m00*d_m00);
        else if(is_node_xmWall(p4est, ni)) w_p00 += -1.0/(d_p00*d_p00);
        else                               w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

        if(is_node_ymWall(p4est, ni))      w_0p0 += -1.0/(d_0p0*d_0p0);
        else if(is_node_ypWall(p4est, ni)) w_0m0 += -1.0/(d_0m0*d_0m0);
        else                               w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

        if(is_node_ypWall(p4est, ni))      w_0m0 += -1.0/(d_0m0*d_0m0);
        else if(is_node_ymWall(p4est, ni)) w_0p0 += -1.0/(d_0p0*d_0p0);
        else                               w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

        //---------------------------------------------------------------------
        // addition to diagonal elements
        //---------------------------------------------------------------------
        if (variable_mu) {

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

        } else {

          if(!is_interface_m00 && !is_node_xmWall(p4est, ni)) {
            w_m00_mm = mu_*w_m00*d_m00_p0/(d_m00_m0+d_m00_p0);
            w_m00_pm = mu_*w_m00*d_m00_m0/(d_m00_m0+d_m00_p0);
            w_m00 = w_m00_mm + w_m00_pm;
          } else {
            w_m00 *= mu_;
          }

          if(!is_interface_p00 && !is_node_xpWall(p4est, ni)) {
            w_p00_mm = mu_*w_p00*d_p00_p0/(d_p00_m0+d_p00_p0);
            w_p00_pm = mu_*w_p00*d_p00_m0/(d_p00_m0+d_p00_p0);
            w_p00    = w_p00_mm + w_p00_pm;
          } else {
            w_p00 *= mu_;
          }

          if(!is_interface_0m0 && !is_node_ymWall(p4est, ni)) {
            w_0m0_mm = mu_*w_0m0*d_0m0_p0/(d_0m0_m0+d_0m0_p0);
            w_0m0_pm = mu_*w_0m0*d_0m0_m0/(d_0m0_m0+d_0m0_p0);
            w_0m0 = w_0m0_mm + w_0m0_pm;
          } else {
            w_0m0 *= mu_;
          }

          if(!is_interface_0p0 && !is_node_ypWall(p4est, ni)) {
            w_0p0_mm = mu_*w_0p0*d_0p0_p0/(d_0p0_m0+d_0p0_p0);
            w_0p0_pm = mu_*w_0p0*d_0p0_m0/(d_0p0_m0+d_0p0_p0);
            w_0p0 = w_0p0_mm + w_0p0_pm;
          } else {
            w_0p0 *= mu_;
          }

        }
#endif

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------

#ifdef P4_TO_P8
        double w_000  = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
#else
        double w_000  = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 );
#endif
//        w_m00 /= w_000; w_p00 /= w_000;
//        w_0m0 /= w_000; w_0p0 /= w_000;
//#ifdef P4_TO_P8
//        w_00m /= w_000; w_00p /= w_000;
//#endif

        //---------------------------------------------------------------------
        // add coefficients in the matrix
        //---------------------------------------------------------------------
        if (node_000_g < fixed_value_idx_g){
          fixed_value_idx_l = n;
          fixed_value_idx_g = node_000_g;
        }
        ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
        if(!is_interface_m00 && !is_node_xmWall(p4est, ni)) {
          if (ABS(w_m00_mm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_mm], w_m00_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_m00_pm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_pm], w_m00_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (ABS(w_m00_mp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_mp], w_m00_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_m00_pp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_m00_pp], w_m00_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }

        if(!is_interface_p00 && !is_node_xpWall(p4est, ni)) {
          if (ABS(w_p00_mm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_mm], w_p00_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_p00_pm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_pm], w_p00_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (ABS(w_p00_mp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_mp], w_p00_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_p00_pp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_p00_pp], w_p00_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }

        if(!is_interface_0m0 && !is_node_ymWall(p4est, ni)) {
          if (ABS(w_0m0_mm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_mm], w_0m0_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0m0_pm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_pm], w_0m0_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (ABS(w_0m0_mp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_mp], w_0m0_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0m0_pp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0m0_pp], w_0m0_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }

        if(!is_interface_0p0 && !is_node_ypWall(p4est, ni)) {
          if (ABS(w_0p0_mm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_mm], w_0p0_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0p0_pm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_pm], w_0p0_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (ABS(w_0p0_mp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_mp], w_0p0_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0p0_pp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_0p0_pp], w_0p0_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
#endif
        }
#ifdef P4_TO_P8
        if(!is_interface_00m && !is_node_zmWall(p4est, ni)) {
          if (ABS(w_00m_mm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_mm], w_00m_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00m_pm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_pm], w_00m_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00m_mp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_mp], w_00m_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00m_pp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00m_pp], w_00m_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
        }

        if(!is_interface_00p && !is_node_zpWall(p4est, ni)) {
          if (ABS(w_00p_mm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_mm], w_00p_mm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00p_pm) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_pm], w_00p_pm/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00p_mp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_mp], w_00p_mp/w_000, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00p_pp) > EPS) {ierr = MatSetValue(A, node_000_g, petsc_gloidx[node_00p_pp], w_00p_pp/w_000, ADD_VALUES); CHKERRXX(ierr);}
        }
#endif
        scalling[n] = w_000;

        if(add_p[n] > 0) matrix_has_nullspace = false;
        continue;
      }

      // if ngbd is crossed and neumman BC
      // then use finite volume method
      // only work if the mesh is uniform close to the interface

      // FIXME: the neumann BC on the interface works only if the interface doesn't touch the edge of the domain
      if (is_ngbd_crossed_neumann && (bc_->interfaceType() == NEUMANN || bc_->interfaceType() == ROBIN))
      {
        double volume_cut_cell = 0.;
        double interface_area  = 0.;

#ifdef P4_TO_P8
        Cube3 cube;
        OctValue  phi_cube;

        for (short k = 0; k < fv_size_z; ++k)
          for (short j = 0; j < fv_size_y; ++j)
            for (short i = 0; i < fv_size_x; ++i)
            {
              cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
              cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];
              cube.z0 = fv_z[k]; cube.z1 = fv_z[k+1];

              phi_cube.val000 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
              phi_cube.val100 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
              phi_cube.val010 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
              phi_cube.val110 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];
              phi_cube.val001 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
              phi_cube.val101 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
              phi_cube.val011 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
              phi_cube.val111 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];

              volume_cut_cell += cube.area_In_Negative_Domain(phi_cube);
              interface_area  += cube.interface_Length_In_Cell(phi_cube);
            }
#else
        Cube2 cube;
        QuadValue phi_cube;

        for (short j = 0; j < fv_size_y; ++j)
          for (short i = 0; i < fv_size_x; ++i)
          {
            cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
            cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];

            phi_cube.val00 = phi_fv[(j+0)*fv_nx + (i+0)];
            phi_cube.val10 = phi_fv[(j+0)*fv_nx + (i+1)];
            phi_cube.val01 = phi_fv[(j+1)*fv_nx + (i+0)];
            phi_cube.val11 = phi_fv[(j+1)*fv_nx + (i+1)];

            volume_cut_cell += cube.area_In_Negative_Domain(phi_cube);
            interface_area  += cube.interface_Length_In_Cell(phi_cube);
          }
#endif

        if (volume_cut_cell>eps*eps)
        {

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

          Cube2 c2;
          QuadValue qv;

          double s_m00 = 0, s_p00 = 0;
          for (short k = 0; k < fv_size_z; ++k)
            for (short j = 0; j < fv_size_y; ++j) {

              c2.x0 = fv_y[j]; c2.x1 = fv_y[j+1];
              c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

              int i = 0;
              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

              s_m00 += c2.area_In_Negative_Domain(qv);

              i = fv_size_x;
              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

              s_p00 += c2.area_In_Negative_Domain(qv);
            }

          double s_0m0 = 0, s_0p0 = 0;
          for (short k = 0; k < fv_size_z; ++k)
            for (short i = 0; i < fv_size_x; ++i) {

              c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
              c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

              int j = 0;
              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

              s_0m0 += c2.area_In_Negative_Domain(qv);

              j = fv_size_y;
              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

              s_0p0 += c2.area_In_Negative_Domain(qv);
            }

          double s_00m = 0, s_00p = 0;
          for (short j = 0; j < fv_size_j; ++j)
            for (short i = 0; i < fv_size_x; ++i) {

              c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
              c2.y0 = fv_y[j]; c2.y1 = fv_y[j+1];

              int k = 0;
              qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
              qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
              qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
              qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

              s_00m += c2.area_In_Negative_Domain(qv);

              k = fv_size_z;
              qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
              qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
              qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
              qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

              s_00p += c2.area_In_Negative_Domain(qv);
            }
#else
          PetscInt node_m00_g = petsc_gloidx[qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm];
          PetscInt node_p00_g = petsc_gloidx[qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm];
          PetscInt node_0m0_g = petsc_gloidx[qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm];
          PetscInt node_0p0_g = petsc_gloidx[qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm];


          double s_m00 = 0, s_p00 = 0;
          for (short j = 0; j < fv_size_y; ++j) {
            int i;
            i = 0;          s_m00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
            i = fv_size_x;  s_p00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
          }

          double s_0m0 = 0, s_0p0 = 0;
          for (short i = 0; i < fv_size_x; ++i) {
            int j;
            j = 0;          s_0m0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
            j = fv_size_y;  s_0p0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
          }

//          double fxx,fyy;
//          fxx = phi_xx_p[n];
//          fyy = phi_yy_p[n];
//          s_0m0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_pmm, fxx, fxx, dx_min);
//          s_0p0 = dx_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mpm, P_ppm, fxx, fxx, dx_min);
//          s_m00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_mmm, P_mpm, fyy, fyy, dy_min);
//          s_p00 = dy_min * fraction_Interval_Covered_By_Irregular_Domain_using_2nd_Order_Derivatives(P_pmm, P_ppm, fyy, fyy, dy_min);

#endif

          /* the code above should return:
           * volume_cut_cell
           * interface_area
           * s_m00, s_p00, ...
           */

          double w_m00 = -.5*(mue_000+mue_m00) * s_m00/dx_min;
          double w_p00 = -.5*(mue_000+mue_p00) * s_p00/dx_min;
          double w_0m0 = -.5*(mue_000+mue_0m0) * s_0m0/dy_min;
          double w_0p0 = -.5*(mue_000+mue_0p0) * s_0p0/dy_min;
#ifdef P4_TO_P8
          double w_00m = -.5*(mue_000+mue_00m) * s_00m/dz_min;
          double w_00p = -.5*(mue_000+mue_00p) * s_00p/dz_min;
#endif

#ifdef P4_TO_P8
          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
#else
          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0);
#endif
          if (bc_->interfaceType() == ROBIN)
          {
            double xyz_p[P4EST_DIM];
            double normal[P4EST_DIM];

            normal[0] = qnnn.dx_central(phi_p);
            normal[1] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
            normal[2] = qnnn.dz_central(phi_p);
            double norm = sqrt(SQR(normal[0])+SQR(normal[1])+SQR(normal[2]));
#else
            double norm = sqrt(SQR(normal[0])+SQR(normal[1]));
#endif

            for (short dir = 0; dir < P4EST_DIM; ++dir)
            {
              xyz_p[dir] = xyz_C[dir] - phi_p[n]*normal[dir]/norm;

              if      (xyz_p[dir] < xyz_C[dir]-dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]-dxyz_m[dir]+EPS;
              else if (xyz_p[dir] > xyz_C[dir]+dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]+dxyz_m[dir]+EPS;
            }

            double robin_coef_proj = bc_->robinCoef(xyz_p);

            if (fabs(robin_coef_proj) > 0) matrix_has_nullspace = false;

            // FIX this for variable mu

            if (robin_coef_proj*phi_p[n] < 1.)
            {
              w_000 += mue_000*(robin_coef_proj/(1.-phi_p[n]*robin_coef_proj))*interface_area;

//              double beta_proj = bc_->interfaceValue(xyz_p);
//              rhs_p[n] += mue_000*robin_coef_proj*phi_p[n]/(1-robin_coef_proj*phi_p[n]) * interface_area*beta_proj;
            }
            else
            {
              w_000 += mue_000*robin_coef_proj*interface_area;
            }
          }

          w_m00 /= w_000; w_p00 /= w_000;
          w_0m0 /= w_000; w_0p0 /= w_000;
#ifdef P4_TO_P8
          w_00m /= w_000; w_00p /= w_000;
#endif

          ierr = MatSetValue(A, node_000_g, node_000_g, 1.0, ADD_VALUES); CHKERRXX(ierr);
          if (ABS(w_m00) > EPS) {ierr = MatSetValue(A, node_000_g, node_m00_g, w_m00,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_p00) > EPS) {ierr = MatSetValue(A, node_000_g, node_p00_g, w_p00,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0m0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0m0_g, w_0m0,  ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_0p0) > EPS) {ierr = MatSetValue(A, node_000_g, node_0p0_g, w_0p0,  ADD_VALUES); CHKERRXX(ierr);}
#ifdef P4_TO_P8
          if (ABS(w_00m) > EPS) {ierr = MatSetValue(A, node_000_g, node_00m_g, w_00m, ADD_VALUES); CHKERRXX(ierr);}
          if (ABS(w_00p) > EPS) {ierr = MatSetValue(A, node_000_g, node_00p_g, w_00p, ADD_VALUES); CHKERRXX(ierr);}
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
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);

  // restore pointers
  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif

  if (variable_mu) {
    ierr = VecRestoreArray(mue_,    &mue_p   ); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  }
  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);

  // check for null space
  // FIXME: the return value should be checked for errors ...
  MPI_Allreduce(MPI_IN_PLACE, &matrix_has_nullspace, 1, MPI_INT, MPI_LAND, p4est->mpicomm);

//  if (matrix_has_nullspace) {
//    ierr = MatSetOption(A, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE); CHKERRXX(ierr);
//    p4est_gloidx_t fixed_value_idx;
//    MPI_Allreduce(&fixed_value_idx_g, &fixed_value_idx, 1, MPI_LONG_LONG_INT, MPI_MIN, p4est->mpicomm);
//    if (fixed_value_idx_g != fixed_value_idx){ // we are not setting the fixed value
//      fixed_value_idx_l = -1;
//      fixed_value_idx_g = fixed_value_idx;
//    } else {
//      // reset the value
//      ierr = MatZeroRows(A, 1, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
//    }
//  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

}

void my_p4est_poisson_nodes_t::setup_negative_variable_coeff_laplace_rhsvec()
{
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

#ifdef P4_TO_P8
  double eps = 1E-6*d_min*d_min*d_min;
#else
  double eps = 1E-6*d_min*d_min;
#endif

  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p;
  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif

  double *mue_p=NULL, *mue_xx_p=NULL, *mue_yy_p=NULL, *mue_zz_p=NULL;
  if (variable_mu)
  {
    ierr = VecGetArray(mue_,    &mue_p   ); CHKERRXX(ierr);
    ierr = VecGetArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
    ierr = VecGetArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecGetArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  }

  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);
  double *rhs_p;
  ierr = VecGetArray(rhs_,    &rhs_p   ); CHKERRXX(ierr);

  double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
  double mue_000, mue_p00, mue_m00, mue_0m0, mue_0p0;

#ifdef P4_TO_P8
  double phi_00m, phi_00p;
  double mue_00m, mue_00p;
#endif

  if (!variable_mu) {
    mue_000 = mu_; mue_p00 = mu_; mue_m00 = mu_; mue_0m0 = mu_; mue_0p0 = mu_;
#ifdef P4_TO_P8
    mue_00m = mu_; mue_00p = mu_;
#endif
  }

  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);
  phi_interp_local.set_input(phi_p, phi_xx_p, phi_yy_p, quadratic);

  /* Daniil: while working on solidification of alloys I came to realize
   * that due to very irregular structure of solidification fronts (where
   * there constantly exist nodes, which belong to one phase but the vertices
   * of their dual cells belong to the other phase. In such cases simple dual
   * cells consisting of 2 (6 in 3D) simplices don't capture such complex
   * topologies. To aleviate this issue I added an option to use a more detailed
   * dual cells consisting of 8 (48 in 3D) simplices. Clearly, it might increase
   * the computational cost quite significantly, but oh well...
   */

  // data for refined cells
  std::vector<double> phi_fv(pow(2*cube_refinement+1,P4EST_DIM),-1);
  double fv_size_x = 0, fv_nx; std::vector<double> fv_x(2*cube_refinement+1, 0);
  double fv_size_y = 0, fv_ny; std::vector<double> fv_y(2*cube_refinement+1, 0);
#ifdef P4_TO_P8
  double fv_size_z = 0, fv_nz; std::vector<double> fv_z(2*cube_refinement+1, 0);
#endif

  double fv_xmin, fv_xmax;
  double fv_ymin, fv_ymax;
#ifdef P4_TO_P8
  double fv_zmin, fv_zmax;
#endif

  double xyz_C[P4EST_DIM];

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    node_xyz_fr_n(n, p4est, nodes, xyz_C);
    double x_C  = xyz_C[0];
    double y_C  = xyz_C[1];
#ifdef P4_TO_P8
    double z_C  = xyz_C[2];
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
      if(bc_->wallType(xyz_C) == DIRICHLET) {
        rhs_p[n] = bc_strength*bc_->wallValue(xyz_C);
        continue;
      }

      // In case if you want first order neumann at walls. Why is it still a thing anyway?
      if(bc_->wallType(xyz_C) == NEUMANN && neumann_wall_first_order)
      {
        if (is_node_xpWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_m00; continue;}
        if (is_node_xmWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_p00; continue;}
        if (is_node_ypWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_0m0; continue;}
        if (is_node_ymWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_0p0; continue;}
#ifdef P4_TO_P8
        if (is_node_zpWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_00m; continue;}
        if (is_node_zmWall(p4est, ni)) {rhs_p[n] = bc_strength*bc_->wallValue(xyz_C)*d_00p; continue;}
#endif
      }
    }

    {
#ifdef P4_TO_P8
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
      if (variable_mu)
        qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0, mue_00m, mue_00p);
#else
      qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
      if (variable_mu)
        qnnn.ngbd_with_quadratic_interpolation(mue_p, mue_000, mue_m00, mue_p00, mue_0m0, mue_0p0);
#endif

      //---------------------------------------------------------------------
      // interface boundary
      //---------------------------------------------------------------------
      if((ABS(phi_000)<eps && bc_->interfaceType() == DIRICHLET) ){
        rhs_p[n] = bc_strength*bc_->interfaceValue(xyz_C);
        continue;
      }

      //---------------------------------------------------------------------
      // check if finite volume is crossed
      //---------------------------------------------------------------------
      bool is_one_positive = false;
      bool is_one_negative = false;

      if (fabs(phi_000) < 2.*diag_min && (bc_->interfaceType() == ROBIN || bc_->interfaceType() == NEUMANN))
//      if (fabs(phi_000) < 2.*diag_min)
      {
        phi_interp_local.initialize(n);
        // determine dimensions of cube
        fv_size_x = 0;
        fv_size_y = 0;
#ifdef P4_TO_P8
        fv_size_z = 0;
#endif
        if(!is_node_xmWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmin = x_C-0.5*dx_min;} else {fv_xmin = x_C;}
        if(!is_node_xpWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmax = x_C+0.5*dx_min;} else {fv_xmax = x_C;}

        if(!is_node_ymWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymin = y_C-0.5*dy_min;} else {fv_ymin = y_C;}
        if(!is_node_ypWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymax = y_C+0.5*dy_min;} else {fv_ymax = y_C;}
#ifdef P4_TO_P8
        if(!is_node_zmWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmin = z_C-0.5*dz_min;} else {fv_zmin = z_C;}
        if(!is_node_zpWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmax = z_C+0.5*dz_min;} else {fv_zmax = z_C;}
#endif

        if (!use_refined_cube) {
          fv_size_x = 1;
          fv_size_y = 1;
#ifdef P4_TO_P8
          fv_size_z = 1;
#endif
        }

        fv_nx = fv_size_x+1;
        fv_ny = fv_size_y+1;
#ifdef P4_TO_P8
        fv_nz = fv_size_z+1;
#endif

        // get coordinates of cube nodes
        double fv_dx = (fv_xmax-fv_xmin)/ (double)(fv_size_x);
        fv_x[0] = fv_xmin;
        for (short i = 1; i < fv_nx; ++i)
          fv_x[i] = fv_x[i-1] + fv_dx;

        double fv_dy = (fv_ymax-fv_ymin)/ (double)(fv_size_y);
        fv_y[0] = fv_ymin;
        for (short i = 1; i < fv_ny; ++i)
          fv_y[i] = fv_y[i-1] + fv_dy;
#ifdef P4_TO_P8
        double fv_dx = (fv_zmax-fv_zmin)/ (double)(fv_size_z);
        fv_z[0] = fv_zmin;
        for (short i = 1; i < fv_nz; ++i)
          fv_z[i] = fv_z[i-1] + fv_dz;
#endif

        // sample level-set function at cube nodes and check if crossed
#ifdef P4_TO_P8
        for (short k = 0; k < fv_nz; ++k)
#endif
          for (short j = 0; j < fv_ny; ++j)
            for (short i = 0; i < fv_nx; ++i)
            {
#ifdef P4_TO_P8
              int idx = k*fv_nx*fv_ny + j*fv_nx + i;
              phi_fv[idx] = phi_interp(fv_x[i], fv_y[j], fv_z[k]);
#else
              int idx = j*fv_nx + i;
//              phi_fv[idx] = phi_interp(fv_x[i], fv_y[j]);
              phi_fv[idx] = phi_interp_local.interpolate(fv_x[i], fv_y[j]);
#endif
              is_one_positive = is_one_positive || phi_fv[idx] > 0;
              is_one_negative = is_one_negative || phi_fv[idx] < 0;
            }
      }

      bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;

      // far away from the interface
      if(phi_000>0. &&  (!is_ngbd_crossed_neumann || bc_->interfaceType() == DIRICHLET ) && (bc_->interfaceType() != NOINTERFACE)){
        if(bc_->interfaceType()==DIRICHLET)
          rhs_p[n] = bc_strength*bc_->interfaceValue(xyz_C);
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


//#ifdef P4_TO_P8
//        if (!( is_interface_0m0 || is_interface_m00 || is_interface_p00 || is_interface_0p0  || is_interface_00m || is_interface_00p) && !is_node_Wall(p4est, ni))
//#else
//        if (!( is_interface_0m0 || is_interface_m00 || is_interface_p00 || is_interface_0p0 ) && !is_node_Wall(p4est, ni))
//#endif
//        {
//          rhs_p[n] /= scalling[n];
//          continue;
//        }

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

          if (variable_mu) {
            double mxx_000 = mue_xx_p[n];
            double mxx_m00 = qnnn.f_m00_linear(mue_xx_p);
            mue_m00 = mue_000*(1-theta_m00/d_m00) + mue_m00*theta_m00/d_m00 + 0.5*theta_m00*(theta_m00-d_m00)*MINMOD(mxx_m00,mxx_000);
          }

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

          if (variable_mu) {
            double mxx_000 = mue_xx_p[n];
            double mxx_p00 = qnnn.f_p00_linear(mue_xx_p);
            mue_p00 = mue_000*(1-theta_p00/d_p00) + mue_p00*theta_p00/d_p00 + 0.5*theta_p00*(theta_p00-d_p00)*MINMOD(mxx_p00,mxx_000);
          }

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

          if (variable_mu) {
            double myy_000 = mue_yy_p[n];
            double myy_0m0 = qnnn.f_0m0_linear(mue_yy_p);
            mue_0m0 = mue_000*(1-theta_0m0/d_0m0) + mue_0m0*theta_0m0/d_0m0 + 0.5*theta_0m0*(theta_0m0-d_0m0)*MINMOD(myy_0m0,myy_000);
          }

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

          if (variable_mu) {
            double myy_000 = mue_yy_p[n];
            double myy_0p0 = qnnn.f_0p0_linear(mue_yy_p);
            mue_0p0 = mue_000*(1-theta_0p0/d_0p0) + mue_0p0*theta_0p0/d_0p0 + 0.5*theta_0p0*(theta_0p0-d_0p0)*MINMOD(myy_0p0,myy_000);
          }

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

          if (variable_mu) {
            double mzz_000 = mue_zz_p[n];
            double mzz_00m = qnnn.f_00m_linear(mue_zz_p);
            mue_00m = mue_000*(1-theta_00m/d_00m) + mue_00m*theta_00m/d_00m + 0.5*theta_00m*(theta_00m-d_00m)*MINMOD(mzz_00m,mzz_000);
          }

          d_00m = theta_00m;

          val_interface_00m = bc_->interfaceValue(x_C, y_C , z_C - theta_00m);
        }
        if( is_interface_00p){
          double phizz_00p = qnnn.f_00p_linear(phi_zz_p);
          double theta_00p = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000, phi_00p, phizz_C, phizz_00p);
          if (theta_00p<eps) theta_00p = eps; if (theta_00p>d_00p) theta_00p = d_00p;
          d_00p_m0 = d_00p_p0 = d_00p_0m = d_00p_0p = 0;

          if (variable_mu) {
            double mzz_000 = mue_zz_p[n];
            double mzz_00p = qnnn.f_00p_linear(mue_zz_p);
            mue_00p = mue_000*(1-theta_00p/d_00p) + mue_00p*theta_00p/d_00p + 0.5*theta_00p*(theta_00p-d_00p)*MINMOD(mzz_00p,mzz_000);
          }

          d_00p = theta_00p;

          val_interface_00p = bc_->interfaceValue(x_C, y_C , z_C + theta_00p);
        }
#endif

        if (use_pointwise_dirichlet)
        {
          std::vector<double> val_interface(2*P4EST_DIM, 0);
          for (short i = 0; i < pointwise_bc[n].size(); ++i)
          {
            interface_point_t *pnt = &pointwise_bc[n][i];
            val_interface[pnt->dir] = pnt->value;
          }

          val_interface_m00 = val_interface[0];
          val_interface_p00 = val_interface[1];
          val_interface_0m0 = val_interface[2];
          val_interface_0p0 = val_interface[3];
#ifdef P4_TO_P8
          val_interface_00m = val_interface[4];
          val_interface_00p = val_interface[5];
#endif
        }

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
        double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0;
#ifdef P4_TO_P8
        double w_00m=0, w_00p=0;
#endif
        if(is_node_xmWall(p4est, ni))      w_p00 += -1.0*wi/(d_p00*d_p00);
        else if(is_node_xpWall(p4est, ni)) w_m00 += -1.0*wi/(d_m00*d_m00);
        else                               w_m00 += -2.0*wi/d_m00/(d_m00+d_p00);

        if(is_node_xpWall(p4est, ni))      w_m00 += -1.0*wi/(d_m00*d_m00);
        else if(is_node_xmWall(p4est, ni)) w_p00 += -1.0*wi/(d_p00*d_p00);
        else                               w_p00 += -2.0*wi/d_p00/(d_m00+d_p00);

        if(is_node_ymWall(p4est, ni))      w_0p0 += -1.0*wj/(d_0p0*d_0p0);
        else if(is_node_ypWall(p4est, ni)) w_0m0 += -1.0*wj/(d_0m0*d_0m0);
        else                               w_0m0 += -2.0*wj/d_0m0/(d_0m0+d_0p0);

        if(is_node_ypWall(p4est, ni))      w_0m0 += -1.0*wj/(d_0m0*d_0m0);
        else if(is_node_ymWall(p4est, ni)) w_0p0 += -1.0*wj/(d_0p0*d_0p0);
        else                               w_0p0 += -2.0*wj/d_0p0/(d_0m0+d_0p0);

#ifdef P4_TO_P8
        if(is_node_zmWall(p4est, ni))      w_00p += -1.0*wk/(d_00p*d_00p);
        else if(is_node_zpWall(p4est, ni)) w_00m += -1.0*wk/(d_00m*d_00m);
        else                               w_00m += -2.0*wk/d_00m/(d_00m+d_00p);

        if(is_node_zpWall(p4est, ni))      w_00m += -1.0*wk/(d_00m*d_00m);
        else if(is_node_zmWall(p4est, ni)) w_00p += -1.0*wk/(d_00p*d_00p);
        else                               w_00p += -2.0*wk/d_00p/(d_00m+d_00p);
#endif
        w_m00 *= 0.5*(mue_000 + mue_m00);
        w_p00 *= 0.5*(mue_000 + mue_p00);
        w_0m0 *= 0.5*(mue_000 + mue_0m0);
        w_0p0 *= 0.5*(mue_000 + mue_0p0);
#ifdef P4_TO_P8
        w_00m *= 0.5*(mue_000 + mue_00m);
        w_00p *= 0.5*(mue_000 + mue_00p);
#endif

        //---------------------------------------------------------------------
        // diag scaling
        //---------------------------------------------------------------------
#ifdef P4_TO_P8
        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );
#else
        double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 );
#endif

//        w_m00 /= w_000; w_p00 /= w_000;
//        w_0m0 /= w_000; w_0p0 /= w_000;
//#ifdef P4_TO_P8
//        w_00m /= w_000; w_00p /= w_000;
//#endif

        //---------------------------------------------------------------------
        // add coefficients to the right hand side
        //---------------------------------------------------------------------
//        if(is_interface_m00) rhs_p[n] -= w_m00 * val_interface_m00;
//        if(is_interface_p00) rhs_p[n] -= w_p00 * val_interface_p00;
//        if(is_interface_0m0) rhs_p[n] -= w_0m0 * val_interface_0m0;
//        if(is_interface_0p0) rhs_p[n] -= w_0p0 * val_interface_0p0;
//#ifdef P4_TO_P8
//        if(is_interface_00m) rhs_p[n] -= w_00m * val_interface_00m;
//        if(is_interface_00p) rhs_p[n] -= w_00p * val_interface_00p;
//#endif
        // FIX this for variable mu
#ifdef P4_TO_P8
        double eps_x = is_node_xmWall(p4est, ni) ? 2*EPS : (is_node_xpWall(p4est, ni) ? -2*EPS : 0);
        double eps_y = is_node_ymWall(p4est, ni) ? 2*EPS : (is_node_ypWall(p4est, ni) ? -2*EPS : 0);
        double eps_z = is_node_zmWall(p4est, ni) ? 2*EPS : (is_node_zpWall(p4est, ni) ? -2*EPS : 0);

        if(is_node_xmWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C, y_C+eps_y, z_C+eps_z) / d_p00;
        else if(is_interface_m00)     rhs_p[n] -= w_m00 * val_interface_m00;

        if(is_node_xpWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C, y_C+eps_y, z_C+eps_z) / d_m00;
        else if(is_interface_p00)     rhs_p[n] -= w_p00 * val_interface_p00;

        if(is_node_ymWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C, z_C+eps_z) / d_0p0;
        else if(is_interface_0m0)     rhs_p[n] -= w_0m0 * val_interface_0m0;

        if(is_node_ypWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C, z_C+eps_z) / d_0m0;
        else if(is_interface_0p0)     rhs_p[n] -= w_0p0 * val_interface_0p0;

        if(is_node_zmWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C+eps_y, z_C) / d_00p;
        else if(is_interface_00m)     rhs_p[n] -= w_00m * val_interface_00m;

        if(is_node_zpWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C+eps_y, z_C) / d_00m;
        else if(is_interface_00p)     rhs_p[n] -= w_00p * val_interface_00p;
#else

        double eps_x = is_node_xmWall(p4est, ni) ? 2*EPS : (is_node_xpWall(p4est, ni) ? -2*EPS : 0);
        double eps_y = is_node_ymWall(p4est, ni) ? 2*EPS : (is_node_ypWall(p4est, ni) ? -2*EPS : 0);

//        double eps_x = 0;
//        double eps_y = 0;

        if(is_node_xmWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C, y_C+eps_y) / d_p00;
        else
          if(is_interface_m00)     rhs_p[n] -= w_m00*val_interface_m00;

        if(is_node_xpWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C, y_C+eps_y) / d_m00;
        else
          if(is_interface_p00)     rhs_p[n] -= w_p00*val_interface_p00;

        if(is_node_ymWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C) / d_0p0;
        else
          if(is_interface_0m0)     rhs_p[n] -= w_0m0*val_interface_0m0;

        if(is_node_ypWall(p4est, ni)) rhs_p[n] += 2.*mue_000*bc_->wallValue(x_C+eps_x, y_C) / d_0m0;
        else
          if(is_interface_0p0)     rhs_p[n] -= w_0p0*val_interface_0p0;
#endif

        rhs_p[n] /= w_000;
        continue;
      }

      // if ngbd is crossed and neumman BC
      // then use finite volume method
      // only work if the mesh is uniform close to the interface

      // FIXME: the neumann BC on the interface works only if the interface doesn't touch the edge of the domain
      if (is_ngbd_crossed_neumann && (bc_->interfaceType() == NEUMANN || bc_->interfaceType() == ROBIN) )
      {
        double volume_cut_cell = 0.;
        double interface_area  = 0.;

#ifdef P4_TO_P8
        Cube3 cube;
        OctValue  phi_cube;

        for (short k = 0; k < fv_size_z; ++k)
          for (short j = 0; j < fv_size_y; ++j)
            for (short i = 0; i < fv_size_x; ++i)
            {
              cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
              cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];
              cube.z0 = fv_z[k]; cube.z1 = fv_z[k+1];

              phi_cube.val000 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
              phi_cube.val100 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
              phi_cube.val010 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
              phi_cube.val110 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];
              phi_cube.val001 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
              phi_cube.val101 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
              phi_cube.val011 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
              phi_cube.val111 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];

              volume_cut_cell += cube.area_In_Negative_Domain(phi_cube);
              interface_area  += cube.interface_Length_In_Cell(phi_cube);
            }
#else
        Cube2 cube;
        QuadValue phi_cube;

        for (short j = 0; j < fv_size_y; ++j)
          for (short i = 0; i < fv_size_x; ++i)
          {
            cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
            cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];

            phi_cube.val00 = phi_fv[(j+0)*fv_nx + (i+0)];
            phi_cube.val10 = phi_fv[(j+0)*fv_nx + (i+1)];
            phi_cube.val01 = phi_fv[(j+1)*fv_nx + (i+0)];
            phi_cube.val11 = phi_fv[(j+1)*fv_nx + (i+1)];

            volume_cut_cell += cube.area_In_Negative_Domain(phi_cube);
            interface_area  += cube.interface_Length_In_Cell(phi_cube);
          }
#endif

        if (volume_cut_cell>eps*eps)
        {
#ifdef P4_TO_P8
          Cube2 c2;
          QuadValue qv;

          double s_m00 = 0, s_p00 = 0;
          for (short k = 0; k < fv_size_z; ++k)
            for (short j = 0; j < fv_size_y; ++j) {

              c2.x0 = fv_y[j]; c2.x1 = fv_y[j+1];
              c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

              int i = 0;
              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

              s_m00 += c2.area_In_Negative_Domain(qv);

              i = fv_size_x;
              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

              s_p00 += c2.area_In_Negative_Domain(qv);
            }

          double s_0m0 = 0, s_0p0 = 0;
          for (short k = 0; k < fv_size_z; ++k)
            for (short i = 0; i < fv_size_x; ++i) {

              c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
              c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

              int j = 0;
              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

              s_0m0 += c2.area_In_Negative_Domain(qv);

              j = fv_size_y;
              qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
              qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
              qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
              qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

              s_0p0 += c2.area_In_Negative_Domain(qv);
            }

          double s_00m = 0, s_00p = 0;
          for (short j = 0; j < fv_size_j; ++j)
            for (short i = 0; i < fv_size_x; ++i) {

              c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
              c2.y0 = fv_y[j]; c2.y1 = fv_y[j+1];

              int k = 0;
              qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
              qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
              qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
              qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

              s_00m += c2.area_In_Negative_Domain(qv);

              k = fv_size_z;
              qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
              qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
              qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
              qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

              s_00p += c2.area_In_Negative_Domain(qv);
            }
#else
          double s_m00 = 0, s_p00 = 0;
          for (short j = 0; j < fv_size_y; ++j) {
            int i;
            i = 0;          s_m00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
            i = fv_size_x;  s_p00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
          }

          double s_0m0 = 0, s_0p0 = 0;
          for (short i = 0; i < fv_size_x; ++i) {
            int j;
            j = 0;          s_0m0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
            j = fv_size_y;  s_0p0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
          }
#endif


          double w_m00 = -0.5*(mue_000 + mue_m00)*s_m00/dx_min;
          double w_p00 = -0.5*(mue_000 + mue_p00)*s_p00/dx_min;
          double w_0m0 = -0.5*(mue_000 + mue_0m0)*s_0m0/dy_min;
          double w_0p0 = -0.5*(mue_000 + mue_0p0)*s_0p0/dy_min;
#ifdef P4_TO_P8
          double w_00m = -0.5*(mue_000 + mue_00m)*s_00m/dz_min;
          double w_00p = -0.5*(mue_000 + mue_00p)*s_00p/dz_min;
          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p);
#else
          double w_000 = add_p[n]*volume_cut_cell - (w_m00 + w_p00 + w_0m0 + w_0p0);
#endif
          rhs_p[n] *= volume_cut_cell;

          if (bc_->interfaceType() == ROBIN)
          {
            double xyz_p[P4EST_DIM];
            double normal[P4EST_DIM];

            normal[0] = qnnn.dx_central(phi_p);
            normal[1] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
            normal[2] = qnnn.dz_central(phi_p);
            double norm = sqrt(SQR(normal[0])+SQR(normal[1])+SQR(normal[2]));
#else
            double norm = sqrt(SQR(normal[0])+SQR(normal[1]));
#endif

            for (short dir = 0; dir < P4EST_DIM; ++dir)
            {
              xyz_p[dir] = xyz_C[dir] - phi_p[n]*normal[dir]/norm;

              if      (xyz_p[dir] < xyz_C[dir]-dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]-dxyz_m[dir]+EPS;
              else if (xyz_p[dir] > xyz_C[dir]+dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]+dxyz_m[dir]+EPS;
            }

            double robin_coef_proj = bc_->robinCoef(xyz_p);

            if (fabs(robin_coef_proj) > 0) matrix_has_nullspace = false;

            // FIX this for variable mu
            if (robin_coef_proj*phi_p[n] < 1.)
            {
              w_000 += mue_000*(robin_coef_proj/(1.-phi_p[n]*robin_coef_proj))*interface_area;

              double beta_proj = bc_->interfaceValue(xyz_p);
              rhs_p[n] += mue_000*robin_coef_proj*phi_p[n]/(1.-robin_coef_proj*phi_p[n]) * interface_area*beta_proj;
            }
            else
            {
              w_000 += mue_000*robin_coef_proj*interface_area;
            }
          }

          double integral_bc = 0;
#ifdef P4_TO_P8
          Cube3 cube;
          OctValue  phi_cube;

          for (short k = 0; k < fv_size_z; ++k)
            for (short j = 0; j < fv_size_y; ++j)
              for (short i = 0; i < fv_size_x; ++i)
              {
                cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
                cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];
                cube.z0 = fv_z[k]; cube.z1 = fv_z[k+1];

                phi_cube.val000 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
                phi_cube.val100 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
                phi_cube.val010 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
                phi_cube.val110 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];
                phi_cube.val001 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
                phi_cube.val101 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
                phi_cube.val011 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
                phi_cube.val111 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];

                integral_bc += cube.integrate_Over_Interface(bc_->getInterfaceValue(), phi_cube);
              }
#else
          Cube2 cube;
          QuadValue phi_cube;

          for (short j = 0; j < fv_size_y; ++j)
            for (short i = 0; i < fv_size_x; ++i)
            {
              cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
              cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];

              phi_cube.val00 = phi_fv[(j+0)*fv_nx + (i+0)];
              phi_cube.val10 = phi_fv[(j+0)*fv_nx + (i+1)];
              phi_cube.val01 = phi_fv[(j+1)*fv_nx + (i+0)];
              phi_cube.val11 = phi_fv[(j+1)*fv_nx + (i+1)];

//              const CF_2 *fff = &bc_->getRobinCoef();
              integral_bc += cube.integrate_Over_Interface(bc_->getInterfaceValue(), phi_cube);
            }
#endif

          rhs_p[n] += mue_000*integral_bc;
          rhs_p[n] /= w_000;

        } else {
          rhs_p[n] = 0.;
        }
      }
    }
  }

//  if (matrix_has_nullspace && fixed_value_idx_l >= 0){
//    rhs_p[fixed_value_idx_l] = 0;
//  }


  // restore pointers
  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif

  if (variable_mu) {
    ierr = VecRestoreArray(mue_,    &mue_p   ); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_xx_, &mue_xx_p); CHKERRXX(ierr);
    ierr = VecRestoreArray(mue_yy_, &mue_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecRestoreArray(mue_zz_, &mue_zz_p); CHKERRXX(ierr);
#endif
  }
  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);

  ierr = VecRestoreArray(rhs_,    &phi_p   ); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_rhsvec_setup, rhs_, 0, 0, 0); CHKERRXX(ierr);
}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_t::set_phi(Vec phi, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
void my_p4est_poisson_nodes_t::set_phi(Vec phi, Vec phi_xx, Vec phi_yy)
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
    if(phi_xx_ != NULL) { ierr = VecDestroy(phi_xx_); CHKERRXX(ierr); }
    if(phi_yy_ != NULL) { ierr = VecDestroy(phi_yy_); CHKERRXX(ierr); }
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_xx_); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &phi_yy_); CHKERRXX(ierr);
#ifdef P4_TO_P8
    if(phi_zz_ != NULL) { ierr = VecDestroy(phi_zz_); CHKERRXX(ierr); }
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
  phi_interp.set_input(phi_, phi_xx_, phi_yy_, phi_zz_, quadratic);
#else
  phi_interp.set_input(phi_, phi_xx_, phi_yy_, quadratic);
#endif
}

#ifdef P4_TO_P8
void my_p4est_poisson_nodes_t::set_mu(Vec mue, Vec mue_xx, Vec mue_yy, Vec mue_zz)
#else
void my_p4est_poisson_nodes_t::set_mu(Vec mue, Vec mue_xx, Vec mue_yy)
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

void my_p4est_poisson_nodes_t::shift_to_exact_solution(Vec sol, Vec uex){
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

void my_p4est_poisson_nodes_t::assemble_matrix(Vec solution)
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
    ierr = VecDuplicate(solution, &phi_); CHKERRXX(ierr);

    Vec tmp;
    ierr = VecGhostGetLocalForm(phi_, &tmp); CHKERRXX(ierr);
    ierr = VecSet(tmp, -1.); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi_, &tmp); CHKERRXX(ierr);
//    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi_); CHKERRXX(ierr);
    set_phi(phi_);
  }

  matrix_has_nullspace = true;
  setup_negative_variable_coeff_laplace_matrix();
  is_matrix_computed = true;
  new_pc = true;

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

    ierr = VecDestroy(phi_xx_); CHKERRXX(ierr);
    phi_xx_ = NULL;

    ierr = VecDestroy(phi_yy_); CHKERRXX(ierr);
    phi_yy_ = NULL;

#ifdef P4_TO_P8
    ierr = VecDestroy(phi_zz_); CHKERRXX(ierr);
    phi_zz_ = NULL;
#endif
  }
}

void my_p4est_poisson_nodes_t::assemble_jump_rhs(Vec rhs_out, CF_2& jump_u, CF_2& jump_un, CF_2& rhs_m, CF_2& rhs_p)
{

  // set local add if none was given
  bool local_add = false;
  if(add_ == NULL)
  {
    local_add = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &add_); CHKERRXX(ierr);
    ierr = VecSet(add_, diag_add_); CHKERRXX(ierr);
  }

  if(phi_ == NULL)
    throw std::domain_error("[CASL_ERROR]: no interface to impose jump conditions on.");

  if (variable_mu)
    throw std::domain_error("[CASL_ERROR]: simple jump solver works only for const mu at the moment.");

  double *phi_p, *phi_xx_p, *phi_yy_p, *add_p;
  ierr = VecGetArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecGetArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  double *phi_zz_p;
  ierr = VecGetArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif

  ierr = VecGetArray(add_,    &add_p   ); CHKERRXX(ierr);

  double *rhs_out_p;
  ierr = VecGetArray(rhs_out, &rhs_out_p); CHKERRXX(ierr);

  for (size_t n = 0; n < nodes->indep_nodes.elem_count; ++n)
    rhs_out_p[n] = 0;

  my_p4est_interpolation_nodes_local_t phi_interp_local(node_neighbors_);
  phi_interp_local.set_input(phi_p, phi_xx_p, phi_yy_p, quadratic);

  /* Daniil: while working on solidification of alloys I came to realize
   * that due to very irregular structure of solidification fronts (where
   * there constantly exist nodes, which belong to one phase but the vertices
   * of their dual cells belong to the other phase. In such cases simple dual
   * cells consisting of 2 (6 in 3D) simplices don't capture such complex
   * topologies. To aleviate this issue I added an option to use a more detailed
   * dual cells consisting of 8 (48 in 3D) simplices. Clearly, it might increase
   * the computational cost quite significantly, but oh well...
   */

  // data for refined cells
  std::vector<double> phi_fv(pow(2*cube_refinement+1,P4EST_DIM),-1);
  double fv_size_x = 0, fv_nx; std::vector<double> fv_x(2*cube_refinement+1, 0);
  double fv_size_y = 0, fv_ny; std::vector<double> fv_y(2*cube_refinement+1, 0);
#ifdef P4_TO_P8
  double fv_size_z = 0, fv_nz; std::vector<double> fv_z(2*cube_refinement+1, 0);
#endif

  double fv_xmin, fv_xmax;
  double fv_ymin, fv_ymax;
#ifdef P4_TO_P8
  double fv_zmin, fv_zmax;
#endif

  double xyz_C[P4EST_DIM];

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);
    node_xyz_fr_n(n, p4est, nodes, xyz_C);
    double x_C  = xyz_C[0];
    double y_C  = xyz_C[1];
#ifdef P4_TO_P8
    double z_C  = xyz_C[2];
#endif

    if (phi_p[n] > 2.*diag_min)

      rhs_out_p[n] += rhs_p(xyz_C[0], xyz_C[1]);
    else if (phi_p[n] < -2.*diag_min)
      rhs_out_p[n] += rhs_m(xyz_C[0], xyz_C[1]);
    else {

      //---------------------------------------------------------------------
      // check if finite volume is crossed
      //---------------------------------------------------------------------
      bool is_one_positive = false;
      bool is_one_negative = false;

      phi_interp_local.initialize(n);
      // determine dimensions of cube
      fv_size_x = 0;
      fv_size_y = 0;
#ifdef P4_TO_P8
      fv_size_z = 0;
#endif
      if(!is_node_xmWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmin = x_C-0.5*dx_min;} else {fv_xmin = x_C;}
      if(!is_node_xpWall(p4est, ni)) {fv_size_x += cube_refinement; fv_xmax = x_C+0.5*dx_min;} else {fv_xmax = x_C;}

      if(!is_node_ymWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymin = y_C-0.5*dy_min;} else {fv_ymin = y_C;}
      if(!is_node_ypWall(p4est, ni)) {fv_size_y += cube_refinement; fv_ymax = y_C+0.5*dy_min;} else {fv_ymax = y_C;}
#ifdef P4_TO_P8
      if(!is_node_zmWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmin = z_C-0.5*dz_min;} else {fv_zmin = z_C;}
      if(!is_node_zpWall(p4est, ni)) {fv_size_z += cube_refinement; fv_zmax = z_C+0.5*dz_min;} else {fv_zmax = z_C;}
#endif

      if (!use_refined_cube) {
        fv_size_x = 1;
        fv_size_y = 1;
#ifdef P4_TO_P8
        fv_size_z = 1;
#endif
      }

      fv_nx = fv_size_x+1;
      fv_ny = fv_size_y+1;
#ifdef P4_TO_P8
      fv_nz = fv_size_z+1;
#endif

      // get coordinates of cube nodes
      double fv_dx = (fv_xmax-fv_xmin)/ (double)(fv_size_x);
      fv_x[0] = fv_xmin;
      for (short i = 1; i < fv_nx; ++i)
        fv_x[i] = fv_x[i-1] + fv_dx;

      double fv_dy = (fv_ymax-fv_ymin)/ (double)(fv_size_y);
      fv_y[0] = fv_ymin;
      for (short i = 1; i < fv_ny; ++i)
        fv_y[i] = fv_y[i-1] + fv_dy;
#ifdef P4_TO_P8
      double fv_dx = (fv_zmax-fv_zmin)/ (double)(fv_size_z);
      fv_z[0] = fv_zmin;
      for (short i = 1; i < fv_nz; ++i)
        fv_z[i] = fv_z[i-1] + fv_dz;
#endif

      // sample level-set function at cube nodes and check if crossed
#ifdef P4_TO_P8
      for (short k = 0; k < fv_nz; ++k)
#endif
        for (short j = 0; j < fv_ny; ++j)
          for (short i = 0; i < fv_nx; ++i)
          {
#ifdef P4_TO_P8
            int idx = k*fv_nx*fv_ny + j*fv_nx + i;
            phi_fv[idx] = phi_interp(fv_x[i], fv_y[j], fv_z[k]);
            phi_fv[idx] = phi_interp_local.interpolate(fv_x[i], fv_y[j], fv_z[k]);
#else
            int idx = j*fv_nx + i;
            //              phi_fv[idx] = phi_interp(fv_x[i], fv_y[j]);
            phi_fv[idx] = phi_interp_local.interpolate(fv_x[i], fv_y[j]);
#endif
            is_one_positive = is_one_positive || phi_fv[idx] > 0;
            is_one_negative = is_one_negative || phi_fv[idx] < 0;
          }


      bool is_ngbd_crossed_neumann = is_one_negative && is_one_positive;

      if (!is_ngbd_crossed_neumann)
      {
        if (phi_p[n] > 0.)  rhs_out_p[n] += rhs_p(xyz_C[0], xyz_C[1]);
        else                rhs_out_p[n] += rhs_m(xyz_C[0], xyz_C[1]);
      } else {

        const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

        double volume_cut_cell = 0.;
        double interface_area  = 0.;
        double integral_bc     = 0.;

#ifdef P4_TO_P8
        p4est_locidx_t n_m00 = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                                : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp) ;
        p4est_locidx_t n_p00 = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                                : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp) ;
        p4est_locidx_t n_0m0 = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                                : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp) ;
        p4est_locidx_t n_0p0 = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                                : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp) ;
        p4est_locidx_t n_00m = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                                : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp) ;
        p4est_locidx_t n_00p = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                                : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp) ;

        double volume_full_cell = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin)*(fv_zmax-fv_zmin);

        Cube3 cube;
        OctValue  phi_cube;

        for (short k = 0; k < fv_size_z; ++k)
          for (short j = 0; j < fv_size_y; ++j)
            for (short i = 0; i < fv_size_x; ++i)
            {
              cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
              cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];
              cube.z0 = fv_z[k]; cube.z1 = fv_z[k+1];

              phi_cube.val000 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
              phi_cube.val100 = phi_fv[(k+0)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
              phi_cube.val010 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
              phi_cube.val110 = phi_fv[(k+0)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];
              phi_cube.val001 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+0)];
              phi_cube.val101 = phi_fv[(k+1)*fv_nx*fv_ny + (j+0)*fv_nx + (i+1)];
              phi_cube.val011 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+0)];
              phi_cube.val111 = phi_fv[(k+1)*fv_nx*fv_ny + (j+1)*fv_nx + (i+1)];

              volume_cut_cell += cube.area_In_Negative_Domain(phi_cube);
              interface_area  += cube.interface_Length_In_Cell(phi_cube);
              integral_bc += cube.integrate_Over_Interface(jump_un, phi_cube);
            }
#else
        p4est_locidx_t n_m00 = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm;
        p4est_locidx_t n_p00 = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm;
        p4est_locidx_t n_0m0 = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm;
        p4est_locidx_t n_0p0 = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm;

        double volume_full_cell = (fv_xmax-fv_xmin)*(fv_ymax-fv_ymin);
        Cube2 cube;
        QuadValue phi_cube;

        for (short j = 0; j < fv_size_y; ++j)
          for (short i = 0; i < fv_size_x; ++i)
          {
            cube.x0 = fv_x[i]; cube.x1 = fv_x[i+1];
            cube.y0 = fv_y[j]; cube.y1 = fv_y[j+1];

            phi_cube.val00 = phi_fv[(j+0)*fv_nx + (i+0)];
            phi_cube.val10 = phi_fv[(j+0)*fv_nx + (i+1)];
            phi_cube.val01 = phi_fv[(j+1)*fv_nx + (i+0)];
            phi_cube.val11 = phi_fv[(j+1)*fv_nx + (i+1)];

            volume_cut_cell += cube.area_In_Negative_Domain(phi_cube);
            interface_area  += cube.interface_Length_In_Cell(phi_cube);
            integral_bc += cube.integrate_Over_Interface(jump_un, phi_cube);
          }

#endif
        double volume_cut_cell_m = volume_cut_cell;
        double volume_cut_cell_p = volume_full_cell - volume_cut_cell;

#ifdef P4_TO_P8
        Cube2 c2;
        QuadValue qv;

        double s_m00 = 0, s_p00 = 0;
        for (short k = 0; k < fv_size_z; ++k)
          for (short j = 0; j < fv_size_y; ++j) {

            c2.x0 = fv_y[j]; c2.x1 = fv_y[j+1];
            c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

            int i = 0;
            qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
            qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
            qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
            qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

            s_m00 += c2.area_In_Negative_Domain(qv);

            i = fv_size_x;
            qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+(j+0)*fv_nx+i];
            qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+(j+1)*fv_nx+i];
            qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+(j+0)*fv_nx+i];
            qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+(j+1)*fv_nx+i];

            s_p00 += c2.area_In_Negative_Domain(qv);
          }

        double s_0m0 = 0, s_0p0 = 0;
        for (short k = 0; k < fv_size_z; ++k)
          for (short i = 0; i < fv_size_x; ++i) {

            c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
            c2.y0 = fv_z[k]; c2.y1 = fv_z[k+1];

            int j = 0;
            qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
            qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
            qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
            qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

            s_0m0 += c2.area_In_Negative_Domain(qv);

            j = fv_size_y;
            qv.val00 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+0)];
            qv.val10 = phi_fv[(k+0)*fv_nx*fv_ny+j*fv_nx+(i+1)];
            qv.val01 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+0)];
            qv.val11 = phi_fv[(k+1)*fv_nx*fv_ny+j*fv_nx+(i+1)];

            s_0p0 += c2.area_In_Negative_Domain(qv);
          }

        double s_00m = 0, s_00p = 0;
        for (short j = 0; j < fv_size_j; ++j)
          for (short i = 0; i < fv_size_x; ++i) {

            c2.x0 = fv_x[i]; c2.x1 = fv_x[i+1];
            c2.y0 = fv_y[j]; c2.y1 = fv_y[j+1];

            int k = 0;
            qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
            qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
            qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
            qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

            s_00m += c2.area_In_Negative_Domain(qv);

            k = fv_size_z;
            qv.val00 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+0)];
            qv.val10 = phi_fv[k*fv_nx*fv_ny+(j+0)*fv_nx+(i+1)];
            qv.val01 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+0)];
            qv.val11 = phi_fv[k*fv_nx*fv_ny+(j+1)*fv_nx+(i+1)];

            s_00p += c2.area_In_Negative_Domain(qv);
          }
#else
        double s_m00 = 0, s_p00 = 0;
        for (short j = 0; j < fv_size_y; ++j) {
          int i;
          i = 0;          s_m00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
          i = fv_size_x;  s_p00 += (fv_y[j+1]-fv_y[j])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[(j+1)*fv_nx+i], dx_min, dy_min);
        }

        double s_0m0 = 0, s_0p0 = 0;
        for (short i = 0; i < fv_size_x; ++i) {
          int j;
          j = 0;          s_0m0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
          j = fv_size_y;  s_0p0 += (fv_x[i+1]-fv_x[i])*fraction_Interval_Covered_By_Irregular_Domain(phi_fv[j*fv_nx+i], phi_fv[j*fv_nx+i+1], dx_min, dy_min);
        }
#endif

        double s_m00_m = s_m00, s_m00_p = (fv_xmax-fv_xmin) - s_m00;
        double s_p00_m = s_p00, s_p00_p = (fv_xmax-fv_xmin) - s_p00;
        double s_0m0_m = s_0m0, s_0m0_p = (fv_ymax-fv_ymin) - s_0m0;
        double s_0p0_m = s_0p0, s_0p0_p = (fv_ymax-fv_ymin) - s_0p0;
#ifdef P4_TO_P8
        double s_00m_m = s_00m, s_00m_p = (fv_zmax-fv_zmin) - s_00m;
        double s_00p_m = s_00p, s_00p_p = (fv_zmax-fv_zmin) - s_00p;
#endif

        double xyz_p[P4EST_DIM];
        double normal[P4EST_DIM];

        normal[0] = qnnn.dx_central(phi_p);
        normal[1] = qnnn.dy_central(phi_p);
#ifdef P4_TO_P8
        normal[2] = qnnn.dz_central(phi_p);
        double norm = sqrt(SQR(normal[0])+SQR(normal[1])+SQR(normal[2]));
#else
        double norm = sqrt(SQR(normal[0])+SQR(normal[1]));
#endif

        for (short dir = 0; dir < P4EST_DIM; ++dir)
        {
          xyz_p[dir] = xyz_C[dir] - phi_p[n]*normal[dir]/norm;

          if      (xyz_p[dir] < xyz_C[dir]-dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]-dxyz_m[dir]+EPS;
          else if (xyz_p[dir] > xyz_C[dir]+dxyz_m[dir]) xyz_p[dir] = xyz_C[dir]+dxyz_m[dir]+EPS;
        }

        double alpha_proj = jump_u(xyz_p[0],xyz_p[1]);
        double beta_proj = jump_un(xyz_p[0],xyz_p[1]);

        double rhs_m_val = rhs_m(xyz_p[0],xyz_p[1]);
        double rhs_p_val = rhs_p(xyz_p[0],xyz_p[1]);

        rhs_out_p[n] += (rhs_m_val*volume_cut_cell_m + rhs_p_val*volume_cut_cell_p + mu_*integral_bc)/volume_full_cell;

        if (phi_p[n] < 0.) {
          double factor = (fabs(phi_p[n])*beta_proj - alpha_proj)/volume_full_cell;

          rhs_out_p[n] -= factor*(add_p[n]*volume_cut_cell_p + mu_*((s_m00_p+s_p00_p)/dx_min+(s_0m0_p+s_0p0_p)/dy_min));
          rhs_out_p[n_m00] += mu_*s_m00_p*factor/dx_min;
          rhs_out_p[n_p00] += mu_*s_p00_p*factor/dx_min;
          rhs_out_p[n_0m0] += mu_*s_0m0_p*factor/dy_min;
          rhs_out_p[n_0p0] += mu_*s_0p0_p*factor/dy_min;
        } else {
          double factor = (fabs(phi_p[n])*beta_proj + alpha_proj)/volume_full_cell;

          rhs_out_p[n] -= factor*(add_p[n]*volume_cut_cell_m + mu_*((s_m00_m+s_p00_m)/dx_min+(s_0m0_m+s_0p0_m)/dy_min));
          rhs_out_p[n_m00] += mu_*s_m00_m*factor/dx_min;
          rhs_out_p[n_p00] += mu_*s_p00_m*factor/dx_min;
          rhs_out_p[n_0m0] += mu_*s_0m0_m*factor/dy_min;
          rhs_out_p[n_0p0] += mu_*s_0p0_m*factor/dy_min;
        }
      }
    }
  }

  // restore pointers
  ierr = VecRestoreArray(phi_,    &phi_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_xx_, &phi_xx_p); CHKERRXX(ierr);
  ierr = VecRestoreArray(phi_yy_, &phi_yy_p); CHKERRXX(ierr);
#ifdef P4_TO_P8
  ierr = VecRestoreArray(phi_zz_, &phi_zz_p); CHKERRXX(ierr);
#endif

  ierr = VecRestoreArray(add_,    &add_p   ); CHKERRXX(ierr);

  ierr = VecRestoreArray(rhs_out, &rhs_out_p   ); CHKERRXX(ierr);

  // get rid of local stuff
  if(local_add)
  {
    ierr = VecDestroy(add_); CHKERRXX(ierr);
    add_ = NULL;
  }

  // update ghosts
  ierr = VecGhostUpdateBegin(rhs_out, ADD_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(rhs_out, ADD_VALUES, SCATTER_REVERSE); CHKERRXX(ierr);

}
