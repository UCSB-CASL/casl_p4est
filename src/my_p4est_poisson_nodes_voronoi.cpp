#ifdef P4_TO_P8
#include "my_p8est_poisson_nodes_voronoi.h"
#include <src/my_p8est_utils.h>
#include <src/my_p8est_refine_coarsen.h>
#include <src/cube3.h>
#include <src/cube2.h>
#else
#include "my_p4est_poisson_nodes_voronoi.h"
#include <src/my_p4est_utils.h>
#include <src/my_p4est_refine_coarsen.h>
#include <src/cube2.h>
#include <src/voronoi2D.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/math.h>

// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_poisson_nodes_voronoi_matrix_preallocation;
extern PetscLogEvent log_my_p4est_poisson_nodes_voronoi_matrix_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_voronoi_rhsvec_setup;
extern PetscLogEvent log_my_p4est_poisson_nodes_voronoi_KSPSolve;
extern PetscLogEvent log_my_p4est_poisson_nodes_voronoi_solve;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

my_p4est_poisson_nodes_voronoi_t::my_p4est_poisson_nodes_voronoi_t(const my_p4est_node_neighbors_t *ngbd)
  : p4est(ngbd->p4est), nodes(ngbd->nodes),
    ghost(ngbd->ghost), myb(ngbd->myb),
    ngbd(ngbd),
    phi_interp(ngbd), robin_coef_interp(ngbd),
    mu(1.), diag_add(0.),
    is_matrix_computed(false), matrix_has_nullspace(false),
    bc(NULL), A(NULL),
    is_phi_dd_owned(false),
    rhs(NULL), phi(NULL), add(NULL),
    robin_coef(NULL)
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

  ::dxyz_min(p4est, dxyz_min);
#ifdef P4_TO_P8
  d_min = MIN(dxyz_min[0], dxyz_min[1], dxyz_min[2]);
  diag_min = sqrt(SQR(dxyz_min[0]) + SQR(dxyz_min[1]) + SQR(dxyz_min[2]));
#else
  d_min = MIN(dxyz_min[0], dxyz_min[1]);
  diag_min = sqrt(SQR(dxyz_min[0]) + SQR(dxyz_min[1]));
#endif

  xyz_min_max(p4est, xyz_min, xyz_max);

  for(int i=0; i<P4EST_DIM; ++i)
    phi_xx[i] = NULL;

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

my_p4est_poisson_nodes_voronoi_t::~my_p4est_poisson_nodes_voronoi_t()
{
  if (A   != NULL) ierr = MatDestroy(A);   CHKERRXX(ierr);
  if (ksp != NULL) ierr = KSPDestroy(ksp); CHKERRXX(ierr);
  if (is_phi_dd_owned){
    for(int i=0; i<P4EST_DIM; ++i)
    {
      if (phi_xx[i] != NULL) ierr = VecDestroy(phi_xx[i]); CHKERRXX(ierr);
    }
  }
}

void my_p4est_poisson_nodes_voronoi_t::preallocate_matrix()
{  
  // enable logging for the preallocation
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_voronoi_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);

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
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

  for (p4est_locidx_t n=0; n<num_owned_local; n++)
  {
    const quad_neighbor_nodes_of_node_t& qnnn = ngbd->get_neighbors(n);

    /*
     * Check for neighboring nodes:
     * 1) If they exist and are local nodes, increment d_nnz[n]
     * 2) If they exist but are not local nodes, increment o_nnz[n]
     * 3) If they do not exist, simply skip
     */

    if (phi_p[n] > 0)
      continue;

    double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
#ifdef P4_TO_P8
    double phi_00m, phi_00p;
#endif

#ifdef P4_TO_P8
    qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
#else
    qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
#endif
    bool is_interface_m00 = phi_m00*phi_000 < 0;
    bool is_interface_p00 = phi_p00*phi_000 < 0;
    bool is_interface_0m0 = phi_0m0*phi_000 < 0;
    bool is_interface_0p0 = phi_0p0*phi_000 < 0;
#ifdef P4_TO_P8
    bool is_interface_00m = phi_00m*phi_000 < 0;
    bool is_interface_00p = phi_00p*phi_000 < 0;
    bool is_interface = is_interface_m00 || is_interface_p00 || is_interface_0m0 || is_interface_0p0 || is_interface_00m || is_interface_00p;
#else
    bool is_interface = is_interface_m00 || is_interface_p00 || is_interface_0m0 || is_interface_0p0;
#endif

    if(is_interface && !bc->interfaceType()==DIRICHLET)
    {
      double xyz_n[P4EST_DIM];
      node_xyz_fr_n(n, p4est, nodes, xyz_n);
      Voronoi2D voro;
      voro.set_Center_Point(xyz_n[0], xyz_n[1]);

      p4est_locidx_t quad_idx;
      p4est_topidx_t tree_idx;

      for(int i=-1; i<2; i+=2)
        for(int j=-1; j<2; j+=2)
        {
          ngbd->find_neighbor_cell_of_node(n, i, j, quad_idx, tree_idx);
          for(int d=0; d<P4EST_CHILDREN; ++d)
          {
            p4est_locidx_t nd = nodes->local_nodes[quad_idx*P4EST_CHILDREN + d];
            if(nd!=n)
            {
              double xnd = node_x_fr_n(nd, p4est, nodes);
              double ynd = node_y_fr_n(nd, p4est, nodes);
              if(phi_p[nd]<0) voro.push(nd, xnd, ynd);
              else            voro.push(nd, xyz_n[0] - 2*(xyz_n[0]-xnd), xyz_n[1] - 2*(xyz_n[1]-ynd));
            }
          }
        }
      voro.enforce_Periodicity(is_periodic(p4est,0), is_periodic(p4est,1), xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1]);
      voro.construct_Partition();

      const std::vector<Point2> *partition;
      voro.get_Partition(partition);
      std::vector<double> phi_values(partition->size());
      for(unsigned int m=0; m<partition->size(); m++)
      {
        phi_values[m] = phi_interp((*partition)[m].x, (*partition)[m].y);
      }
      voro.set_Level_Set_Values(phi_values, phi_000);
      voro.clip_Interface();

      const std::vector<Voronoi2DPoint> *points;
      voro.get_Points(points);
      for(unsigned int m=0; m<points->size(); ++m)
      {
        if((*points)[m].n>=0)
          (*points)[m].n < num_owned_local ? d_nnz[n]++ : o_nnz[n]++;
      }
    }
    else
    {
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
  }

  ierr = VecRestoreArray(phi, &phi_p); CHKERRXX(ierr);

  ierr = MatSeqAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_voronoi_matrix_preallocation, A, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_poisson_nodes_voronoi_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
{
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_voronoi_solve, A, rhs, ksp, 0); CHKERRXX(ierr);

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

  // set local add if none was given
  bool local_add = false;
  if(add == NULL)
  {
    local_add = true;
    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &add); CHKERRXX(ierr);
    ierr = VecSet(add, diag_add); CHKERRXX(ierr);
  }

  // set a local phi if not was given
  bool local_phi = false;
  if(phi == NULL)
  {
    local_phi = true;
    ierr = VecDuplicate(solution, &phi); CHKERRXX(ierr);

    Vec tmp;
    ierr = VecGhostGetLocalForm(phi, &tmp); CHKERRXX(ierr);
    ierr = VecSet(tmp, -1.); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(phi, &tmp); CHKERRXX(ierr);
//    ierr = VecCreateSeq(PETSC_COMM_SELF, nodes->num_owned_indeps, &phi_); CHKERRXX(ierr);
    set_phi(phi);
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

//    // Finally, if matrix has a nullspace, one should _NOT_ use Gaussian-Elimination as the smoother for the coarsest grid
//    if (matrix_has_nullspace){
//      ierr = PetscOptionsSetValue("-pc_hypre_boomeramg_relax_type_coarse", "symmetric-SOR/Jacobi"); CHKERRXX(ierr);
//    }
  }
  ierr = PCSetFromOptions(pc); CHKERRXX(ierr);

  // setup rhs
  setup_negative_laplace_rhsvec();

  // Solve the system
  ierr = KSPSetTolerances(ksp, 1e-14, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRXX(ierr);
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_voronoi_KSPSolve, ksp, rhs, solution, 0); CHKERRXX(ierr);
  ierr = KSPSolve(ksp, rhs, solution); CHKERRXX(ierr);
  ierr = PetscLogEventEnd  (log_my_p4est_poisson_nodes_voronoi_KSPSolve, ksp, rhs, solution, 0); CHKERRXX(ierr);

  // update ghosts
  ierr = VecGhostUpdateBegin(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
  ierr = VecGhostUpdateEnd(solution, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

  // get rid of local stuff
  if(local_add)
  {
    ierr = VecDestroy(add); CHKERRXX(ierr);
    add = NULL;
  }
  if(local_phi)
  {
    ierr = VecDestroy(phi); CHKERRXX(ierr);
    phi = NULL;
    for(int i=0; i<P4EST_DIM; ++i)
    {
      ierr = VecDestroy(phi_xx[i]); CHKERRXX(ierr);
      phi_xx[i] = NULL;
    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_voronoi_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
}



void my_p4est_poisson_nodes_voronoi_t::setup_negative_laplace_matrix()
{
  preallocate_matrix();

  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_voronoi_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);

  double eps = 1E-6*d_min*d_min;

  double *phi_p, *phi_xx_p[P4EST_DIM], *add_p, *robin_coef_p = NULL;
  ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);
  ierr = VecGetArray(add, &add_p); CHKERRXX(ierr);
  for(int i=0; i<P4EST_DIM; ++i)
  {
    ierr = VecGetArray(phi_xx[i], &phi_xx_p[i]); CHKERRXX(ierr);
  }
  if (robin_coef)
  {
    ierr = VecGetArray(robin_coef, &robin_coef_p); CHKERRXX(ierr);
  }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {    
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    double xyz_n[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz_n);

    const quad_neighbor_nodes_of_node_t& qnnn = ngbd->get_neighbors(n);

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

#ifdef P4_TO_P8
    if(is_node_Wall(p4est, ni) && bc->wallType(xyz_n[0],xyz_n[1],xyz_n[2]) == DIRICHLET)
#else
    if(is_node_Wall(p4est, ni) && bc->wallType(xyz_n[0],xyz_n[1]) == DIRICHLET)
#endif
    {
      ierr = MatSetValue(A, node_000_g, node_000_g, 1., ADD_VALUES); CHKERRXX(ierr);
      if (phi_p[n]<0) matrix_has_nullspace = false;
      continue;
    }

    double phi_000, phi_p00, phi_m00, phi_0m0, phi_0p0;
#ifdef P4_TO_P8
    double phi_00m, phi_00p;
#endif

#ifdef P4_TO_P8
    qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0, phi_00m, phi_00p);
#else
    qnnn.ngbd_with_quadratic_interpolation(phi_p, phi_000, phi_m00, phi_p00, phi_0m0, phi_0p0);
#endif
    phi_m00 = qnnn.f_m00_linear(phi_p);
    phi_p00 = qnnn.f_p00_linear(phi_p);
    phi_0m0 = qnnn.f_0m0_linear(phi_p);
    phi_0p0 = qnnn.f_0p0_linear(phi_p);

    //---------------------------------------------------------------------
    // interface boundary
    //---------------------------------------------------------------------
    if((ABS(phi_000)<eps && bc->interfaceType() == DIRICHLET) ){
      ierr = MatSetValue(A, node_000_g, node_000_g, 1., ADD_VALUES); CHKERRXX(ierr);

      matrix_has_nullspace=false;
      continue;
    }

    // far away from the interface
    if(phi_000>0){
      ierr = MatSetValue(A, node_000_g, node_000_g, 1., ADD_VALUES); CHKERRXX(ierr);
      continue;
    }

    bool is_interface_m00 = phi_m00*phi_000 < 0;
    bool is_interface_p00 = phi_p00*phi_000 < 0;
    bool is_interface_0m0 = phi_0m0*phi_000 < 0;
    bool is_interface_0p0 = phi_0p0*phi_000 < 0;
#ifdef P4_TO_P8
    bool is_interface_00m = phi_00m*phi_000 < 0;
    bool is_interface_00p = phi_00p*phi_000 < 0;
    bool is_interface = is_interface_m00 || is_interface_p00 || is_interface_0m0 || is_interface_0p0 || is_interface_00m || is_interface_00p;
#else
    bool is_interface = is_interface_m00 || is_interface_p00 || is_interface_0m0 || is_interface_0p0;
#endif

    /* for the Robin or Neumann cases, do finite volume with voronoi control volume to exclude phi>0 points */
    if(is_interface && !bc->interfaceType()==DIRICHLET)
    {
      Voronoi2D voro;
      voro.set_Center_Point(xyz_n[0], xyz_n[1]);

      p4est_locidx_t quad_idx;
      p4est_topidx_t tree_idx;

      for(int i=-1; i<2; i+=2)
      {
        for(int j=-1; j<2; j+=2)
        {
          ngbd->find_neighbor_cell_of_node(n, i, j, quad_idx, tree_idx);
          for(int d=0; d<P4EST_CHILDREN; ++d)
          {
            p4est_locidx_t nd = nodes->local_nodes[quad_idx*P4EST_CHILDREN + d];
            if(nd!=n)
            {
              double xnd = node_x_fr_n(nd, p4est, nodes);
              double ynd = node_y_fr_n(nd, p4est, nodes);
              if(phi_p[nd]<0) voro.push(nd, xnd, ynd);
              else            voro.push(nd, xyz_n[0] - 2*(xyz_n[0]-xnd), xyz_n[1] - 2*(xyz_n[1]-ynd));
            }
          }
        }
      }

      voro.enforce_Periodicity(is_periodic(p4est,0), is_periodic(p4est,1), xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1]);
      voro.construct_Partition();

      const std::vector<Point2> *partition;
      voro.get_Partition(partition);
      std::vector<double> phi_values(partition->size());
      for(unsigned int m=0; m<partition->size(); m++)
      {
        phi_values[m] = phi_interp((*partition)[m].x, (*partition)[m].y);
      }
      voro.set_Level_Set_Values(phi_values, phi_000);
      voro.clip_Interface();

      /* now assemble the matrix */
      double volume = voro.get_volume();
      ierr = MatSetValue(A, node_000_g, node_000_g, add_p[n]*volume, ADD_VALUES); CHKERRXX(ierr);

      const std::vector<Voronoi2DPoint> *points;
      voro.get_Partition(partition);
      voro.get_Points(points);
      for(unsigned int m=0; m<partition->size(); m++)
      {
#ifdef P4_TO_P8
        double s = (*points)[m].s;
        double d = ((*points)[m].p - Point3(xyz_n[0],xyz_n[1],xyz_n[2])).norm_L2();
#else
        int k = mod(m-1, points->size());
        double s = ((*partition)[m] - (*partition)[k]).norm_L2();
        double d = ((*points)[m].p - Point2(xyz_n[0],xyz_n[1])).norm_L2();
#endif
        double robin_coef_n;

        switch((*points)[m].n)
        {
        case WALL_m00:
        case WALL_p00:
        case WALL_0m0:
        case WALL_0p0:
#ifdef P4_TO_P8
        case WALL_00m:
        case WALL_00p:
#endif
        case INTERFACE:
          switch(bc->interfaceType())
          {
          case ROBIN:
            robin_coef_n = robin_coef_interp(xyz_n[0]-phi_p[n]*qnnn.dx_central(phi_p), xyz_n[1]-phi_p[n]*qnnn.dy_central(phi_p));
            if(1-robin_coef_n*phi_p[n]>0)
            {
              ierr = MatSetValue(A, node_000_g, node_000_g, mu*s*robin_coef_n/(1-robin_coef_n*phi_p[n]), ADD_VALUES); CHKERRXX(ierr);
            }
            else
            {
              ierr = MatSetValue(A, node_000_g, node_000_g, mu*s*robin_coef_p[n], ADD_VALUES); CHKERRXX(ierr);
            }
            break;
          default:
            ;
          }
          break;
        default:
          ierr = MatSetValue(A, node_000_g, node_000_g                  , mu*s/d, ADD_VALUES); CHKERRXX(ierr);
          ierr = MatSetValue(A, node_000_g, petsc_gloidx[(*points)[m].n],-mu*s/d, ADD_VALUES); CHKERRXX(ierr);
        }
      }
    }
    else
    {
      matrix_has_nullspace = false;

      // given boundary condition at interface from quadratic interpolation
      if( is_interface_m00) {
        double phixx_m00 = qnnn.f_m00_linear(phi_xx_p[0]);
        double theta_m00 = interface_Location_With_Second_Order_Derivative(0., d_m00, phi_000, phi_m00, phi_xx_p[0][n], phixx_m00);
        if (theta_m00<eps) theta_m00 = eps; if (theta_m00>d_m00) theta_m00 = d_m00;
        d_m00_m0 = d_m00_p0 = 0;
#ifdef P4_TO_P8
        d_m00_0m = d_m00_0p = 0;
#endif
        d_m00 = theta_m00;
      }
      if( is_interface_p00){
        double phixx_p00 = qnnn.f_p00_linear(phi_xx_p[0]);
        double theta_p00 = interface_Location_With_Second_Order_Derivative(0., d_p00, phi_000, phi_p00, phi_xx_p[0][n], phixx_p00);
        if (theta_p00<eps) theta_p00 = eps; if (theta_p00>d_p00) theta_p00 = d_p00;
        d_p00_m0 = d_p00_p0 = 0;
#ifdef P4_TO_P8
        d_p00_0m = d_p00_0p = 0;
#endif
        d_p00 = theta_p00;
      }
      if( is_interface_0m0){
        double phiyy_0m0 = qnnn.f_0m0_linear(phi_xx_p[1]);
        double theta_0m0 = interface_Location_With_Second_Order_Derivative(0., d_0m0, phi_000, phi_0m0, phi_xx_p[1][n], phiyy_0m0);
        if (theta_0m0<eps) theta_0m0 = eps; if (theta_0m0>d_0m0) theta_0m0 = d_0m0;
        d_0m0_m0 = d_0m0_p0 = 0;
#ifdef P4_TO_P8
        d_0m0_0m = d_0m0_0p = 0;
#endif
        d_0m0 = theta_0m0;
      }
      if( is_interface_0p0){
        double phiyy_0p0 = qnnn.f_0p0_linear(phi_xx_p[1]);
        double theta_0p0 = interface_Location_With_Second_Order_Derivative(0., d_0p0, phi_000, phi_0p0, phi_xx_p[1][n], phiyy_0p0);
        if (theta_0p0<eps) theta_0p0 = eps; if (theta_0p0>d_0p0) theta_0p0 = d_0p0;
        d_0p0_m0 = d_0p0_p0 = 0;
#ifdef P4_TO_P8
        d_0p0_0m = d_0p0_0p = 0;
#endif
        d_0p0 = theta_0p0;
      }
#ifdef P4_TO_P8
      if( is_interface_00m){
        double phizz_00m = qnnn.f_00m_linear(phi_xx_p[2]);
        double theta_00m = interface_Location_With_Second_Order_Derivative(0., d_00m, phi_000, phi_00m, phi_zz_p[2][n], phizz_00m);
        if (theta_00m<eps) theta_00m = eps; if (theta_00m>d_00m) theta_00m = d_00m;
        d_00m_m0 = d_00m_p0 = d_00m_0m = d_00m_0p = 0;
        d_00m = theta_00m;
      }
      if( is_interface_00p){
        double phizz_00p = qnnn.f_00p_linear(phi_xx_p[2]);
        double theta_00p = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000, phi_00p, phi_zz_p[2][n], phizz_00p);
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
      double w_m00=0, w_p00=0, w_0m0=0, w_0p0=0, w_00m=0, w_00p=0;

      if     (is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
      else if(is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
      else                               w_m00 += -2./d_m00/(d_m00+d_p00);

      if     (is_node_xpWall(p4est, ni)) w_m00 += -1./(d_m00*d_m00);
      else if(is_node_xmWall(p4est, ni)) w_p00 += -1./(d_p00*d_p00);
      else                               w_p00 += -2./d_p00/(d_m00+d_p00);

      if     (is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
      else if(is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
      else                               w_0m0 += -2./d_0m0/(d_0m0+d_0p0);

      if     (is_node_ypWall(p4est, ni)) w_0m0 += -1./(d_0m0*d_0m0);
      else if(is_node_ymWall(p4est, ni)) w_0p0 += -1./(d_0p0*d_0p0);
      else                               w_0p0 += -2./d_0p0/(d_0m0+d_0p0);

      if     (is_node_zmWall(p4est, ni)) w_00p += -1./(d_00p*d_00p);
      else if(is_node_zpWall(p4est, ni)) w_00m += -1./(d_00m*d_00m);
      else                               w_00m += -2./d_00m/(d_00m+d_00p);

      if     (is_node_zpWall(p4est, ni)) w_00m += -1./(d_00m*d_00m);
      else if(is_node_zmWall(p4est, ni)) w_00p += -1./(d_00p*d_00p);
      else                               w_00p += -2./d_00p/(d_00m+d_00p);

      w_m00 *= wi * mu; w_p00 *= wi * mu;
      w_0m0 *= wj * mu; w_0p0 *= wj * mu;
      w_00m *= wk * mu; w_00p *= wk * mu;

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

      w_m00 *= weight_on_Dxx*mu;
      w_p00 *= weight_on_Dxx*mu;
      w_0m0 *= weight_on_Dyy*mu;
      w_0p0 *= weight_on_Dyy*mu;

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
    }
  }

  // Assemble the matrix
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
  ierr = MatAssemblyEnd  (A, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);


  // restore pointers
  ierr = VecRestoreArray(phi,    &phi_p   ); CHKERRXX(ierr);
  for(int i=0; i<P4EST_DIM; ++i)
  {
    ierr = VecRestoreArray(phi_xx[i], &phi_xx_p[i]); CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(add,    &add_p   ); CHKERRXX(ierr);
  if (robin_coef) {
    ierr = VecRestoreArray(robin_coef, &robin_coef_p); CHKERRXX(ierr);
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
      ierr = MatZeroRows(A, 0, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
    } else {
      // reset the value
      ierr = MatZeroRows(A, 1, (PetscInt*)(&fixed_value_idx_g), 1.0, NULL, NULL); CHKERRXX(ierr);
    }
  }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_voronoi_matrix_setup, A, 0, 0, 0); CHKERRXX(ierr);
}



void my_p4est_poisson_nodes_voronoi_t::setup_negative_laplace_rhsvec()
{
  // register for logging purpose
  ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_voronoi_rhsvec_setup, 0, 0, 0, 0); CHKERRXX(ierr);

  double eps = 1E-6*d_min*d_min;

  double *phi_p, *phi_xx_p[P4EST_DIM], *add_p, *rhs_p, *robin_coef_p = NULL;
  ierr = VecGetArray(phi,    &phi_p   ); CHKERRXX(ierr);
  for(int i=0; i<P4EST_DIM; ++i)
  {
    ierr = VecGetArray(phi_xx[i], &phi_xx_p[i]); CHKERRXX(ierr);
  }
  ierr = VecGetArray(add,    &add_p   ); CHKERRXX(ierr);
  ierr = VecGetArray(rhs,    &rhs_p   ); CHKERRXX(ierr);
  if (robin_coef) { ierr = VecGetArray(robin_coef, &robin_coef_p); CHKERRXX(ierr); }

  for(p4est_locidx_t n=0; n<nodes->num_owned_indeps; n++) // loop over nodes
  {
    // tree information
    p4est_indep_t *ni = (p4est_indep_t*)sc_array_index(&nodes->indep_nodes, n);

    //---------------------------------------------------------------------
    // Information at neighboring nodes
    //---------------------------------------------------------------------
    double xyz_n[P4EST_DIM];
    node_xyz_fr_n(n, p4est, nodes, xyz_n);

    const quad_neighbor_nodes_of_node_t& qnnn = ngbd->get_neighbors(n);

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

#ifdef P4_TO_P8
    if(is_node_Wall(p4est, ni) && bc->wallType(xyz_n[0],xyz_n[1],xyz_n[2]) == DIRICHLET)
#else
    if(is_node_Wall(p4est, ni) && bc->wallType(xyz_n[0],xyz_n[1]) == DIRICHLET)
#endif
    {
#ifdef P4_TO_P8
      rhs_p[n] = bc->wallValue(xyz_n[0],xyz_n[1],xyz_n[2]);
#else
      rhs_p[n] = bc->wallValue(xyz_n[0],xyz_n[1]);
#endif
      continue;
    }

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
    if((ABS(phi_000)<eps && bc->interfaceType() == DIRICHLET) ){
#ifdef P4_TO_P8
      rhs_p[n] = bc->interfaceValue(xyz_n[0],xyz_n[1],xyz_n[2]);
#else
      rhs_p[n] = bc->interfaceValue(xyz_n[0],xyz_n[1]);
#endif
      continue;
    }

    // far away from the interface
    if(phi_000>0){
      rhs_p[n] = 0;
      continue;
    }

    bool is_interface_m00 = phi_m00*phi_000 < 0;
    bool is_interface_p00 = phi_p00*phi_000 < 0;
    bool is_interface_0m0 = phi_0m0*phi_000 < 0;
    bool is_interface_0p0 = phi_0p0*phi_000 < 0;
#ifdef P4_TO_P8
    bool is_interface_00m = phi_00m*phi_000 < 0;
    bool is_interface_00p = phi_00p*phi_000 < 0;
    bool is_interface = is_interface_m00 || is_interface_p00 || is_interface_0m0 || is_interface_0p0 || is_interface_00m || is_interface_00p;
#else
    bool is_interface = is_interface_m00 || is_interface_p00 || is_interface_0m0 || is_interface_0p0;
#endif

    if(is_interface && !bc->interfaceType()==DIRICHLET)
    {
      Voronoi2D voro;
      voro.set_Center_Point(xyz_n[0], xyz_n[1]);

      p4est_locidx_t quad_idx;
      p4est_topidx_t tree_idx;

      for(int i=-1; i<2; i+=2)
        for(int j=-1; j<2; j+=2)
        {
          ngbd->find_neighbor_cell_of_node(n, i, j, quad_idx, tree_idx);
          for(int d=0; d<P4EST_CHILDREN; ++d)
          {
            p4est_locidx_t nd = nodes->local_nodes[quad_idx*P4EST_CHILDREN + d];
            if(nd!=n)
            {
              double xnd = node_x_fr_n(nd, p4est, nodes);
              double ynd = node_y_fr_n(nd, p4est, nodes);
              if(phi_p[nd]<0) voro.push(nd, xnd, ynd);
              else            voro.push(nd, xyz_n[0] - 2*(xyz_n[0]-xnd), xyz_n[1] - 2*(xyz_n[1]-ynd));
            }
          }
        }

      voro.enforce_Periodicity(is_periodic(p4est,0), is_periodic(p4est,1), xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1]);
      voro.construct_Partition();

      const std::vector<Point2> *partition;
      voro.get_Partition(partition);
      std::vector<double> phi_values(partition->size());
      for(unsigned int m=0; m<partition->size(); m++)
      {
        phi_values[m] = phi_interp((*partition)[m].x, (*partition)[m].y);
      }
      voro.set_Level_Set_Values(phi_values, phi_000);
      voro.clip_Interface();

      /* now update the right hand side */
      double volume = voro.get_volume();
      rhs_p[n] *= volume;

      const std::vector<Voronoi2DPoint> *points;
      voro.get_Partition(partition);
      voro.get_Points(points);
      for(unsigned int m=0; m<partition->size(); m++)
      {
#ifdef P4_TO_P8
        double s = (*points)[m].s;
#else
        int k = mod(m-1, points->size());
        double s = ((*partition)[m] - (*partition)[k]).norm_L2();
#endif
        double robin_coef_n;

        switch((*points)[m].n)
        {
        case WALL_m00:
#ifdef P4_TO_P8
          switch(bc->wallType(xyz_min[0], xyz_n[1], xyz_n[2]))
#else
          switch(bc->wallType(xyz_min[0], xyz_n[1]))
#endif
          {
          case NEUMANN:
#ifdef P4_TO_P8
            rhs_p[n] += mu*s*bc->wallValue(xyz_min[0], xyz_n[1], xyz_n[2]);
#else
            rhs_p[n] += mu*s*bc->wallValue(xyz_min[0], xyz_n[1]);
#endif
            break;
          default:
            throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_nodes_voronoi_t: unknown boundary condition type.");
          }
          break;

        case WALL_p00:
#ifdef P4_TO_P8
          switch(bc->wallType(xyz_max[0], xyz_n[1], xyz_n[2]))
#else
          switch(bc->wallType(xyz_max[0], xyz_n[1]))
#endif
          {
          case NEUMANN:
#ifdef P4_TO_P8
            rhs_p[n] += mu*s*bc->wallValue(xyz_max[0], xyz_n[1], xyz_n[2]);
#else
            rhs_p[n] += mu*s*bc->wallValue(xyz_max[0], xyz_n[1]);
#endif
            break;
          default:
            throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_nodes_voronoi_t: unknown boundary condition type.");
          }
          break;

        case WALL_0m0:
#ifdef P4_TO_P8
          switch(bc->wallType(xyz_n[0], xyz_min[1], xyz_n[2]))
#else
          switch(bc->wallType(xyz_n[0], xyz_min[1]))
#endif
          {
          case NEUMANN:
#ifdef P4_TO_P8
            rhs_p[n] += mu*s*bc->wallValue(xyz_n[0], xyz_min[1], xyz_n[2]);
#else
            rhs_p[n] += mu*s*bc->wallValue(xyz_n[0], xyz_min[1]);
#endif
            break;
          default:
            throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_nodes_voronoi_t: unknown boundary condition type.");
          }
          break;

        case WALL_0p0:
#ifdef P4_TO_P8
          switch(bc->wallType(xyz_n[0], xyz_max[1], xyz_n[2]))
#else
          switch(bc->wallType(xyz_n[0], xyz_max[1]))
#endif
          {
          case NEUMANN:
#ifdef P4_TO_P8
            rhs_p[n] += mu*s*bc->wallValue(xyz_n[0], xyz_max[1], xyz_n[2]);
#else
            rhs_p[n] += mu*s*bc->wallValue(xyz_n[0], xyz_max[1]);
#endif
            break;
          default:
            throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_nodes_voronoi_t: unknown boundary condition type.");
          }
          break;

#ifdef P4_TO_P8
        case WALL_00m:
          switch(bc->wallType(xyz_n[0], xyz_n[1], xyz_min[2]))
          {
          case NEUMANN:
            rhs_p[n] += mu*s*bc->wallValue(xyz_n[0], xyz_n[1], xyz_min[2]);
            break;
          default:
            throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_nodes_voronoi_t: unknown boundary condition type.");
          }
          break;

        case WALL_00p:
          switch(bc->wallType(xyz_n[0], xyz_n[1], xyz_max[2]))
          {
          case NEUMANN:
            rhs_p[n] += mu*s*bc->wallValue(xyz_n[0], xyz_n[1], xyz_max[2]);
            break;
          default:
            throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_nodes_voronoi_t: unknown boundary condition type.");
          }
          break;
#endif

        case INTERFACE:
          switch(bc->interfaceType())
          {
          case NEUMANN:
#ifdef P4_TO_P8
            throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_nodes_voronoi_t: Neumann boundary conditions not implemented in 3D yet ...");
#else
            rhs_p[n] += mu*s*bc->interfaceValue(((*partition)[m].x+(*partition)[k].x)/2, ((*partition)[m].y+(*partition)[k].y)/2);
#endif
            break;
          case ROBIN:
#ifdef P4_TO_P8
            throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_nodes_voronoi_t: Neumann boundary conditions not implemented in 3D yet ...");
#else
            robin_coef_n = robin_coef_interp(xyz_n[0]-phi_p[n]*qnnn.dx_central(phi_p), xyz_n[1]-phi_p[n]*qnnn.dy_central(phi_p));
            if(1-robin_coef_n*phi_p[n]>0) rhs_p[n] += mu/(1-robin_coef_n*phi_p[n]) * s*bc->interfaceValue(((*partition)[m].x+(*partition)[k].x)/2, ((*partition)[m].y+(*partition)[k].y)/2);
            else                             rhs_p[n] += mu*s*bc->interfaceValue(((*partition)[m].x+(*partition)[k].x)/2, ((*partition)[m].y+(*partition)[k].y)/2);
#endif
            break;
          default:
            throw std::invalid_argument("[CASL_ERROR]: my_p4est_poisson_nodes_voronoi_t: unknown boundary condition type.");
          }
          break;
        default:
          ;
        }
      }
    }
    else
    {
      double val_interface_m00 = 0;
      double val_interface_p00 = 0;
      double val_interface_0m0 = 0;
      double val_interface_0p0 = 0;
#ifdef P4_TO_P8
      double val_interface_00m = 0;
      double val_interface_00p = 0;
#endif

      // given boundary condition at interface from quadratic interpolation
      if( is_interface_m00) {
        double phixx_m00 = qnnn.f_m00_linear(phi_xx_p[0]);
        double theta_m00 = interface_Location_With_Second_Order_Derivative(0., d_m00, phi_000, phi_m00, phi_xx_p[0][n], phixx_m00);
        if (theta_m00<eps) theta_m00 = eps; if (theta_m00>d_m00) theta_m00 = d_m00;
        d_m00_m0 = d_m00_p0 = 0;
#ifdef P4_TO_P8
        d_m00_0m = d_m00_0p = 0;
#endif
        d_m00 = theta_m00;
#ifdef P4_TO_P8
        val_interface_m00 = bc->interfaceValue(xyz_n[0] - theta_m00, xyz_n[1], xyz_n[2]);
#else
        val_interface_m00 = bc->interfaceValue(xyz_n[0] - theta_m00, xyz_n[1]);
#endif
      }
      if( is_interface_p00){
        double phixx_p00 = qnnn.f_p00_linear(phi_xx_p[0]);
        double theta_p00 = interface_Location_With_Second_Order_Derivative(0., d_p00, phi_000, phi_p00, phi_xx_p[0][n], phixx_p00);
        if (theta_p00<eps) theta_p00 = eps; if (theta_p00>d_p00) theta_p00 = d_p00;
        d_p00_m0 = d_p00_p0 = 0;
#ifdef P4_TO_P8
        d_p00_0m = d_p00_0p = 0;
#endif
        d_p00 = theta_p00;
#ifdef P4_TO_P8
        val_interface_p00 = bc->interfaceValue(xyz_n[0] + theta_p00, xyz_n[1], xyz_n[2]);
#else
        val_interface_p00 = bc->interfaceValue(xyz_n[0] + theta_p00, xyz_n[1]);
#endif
      }
      if( is_interface_0m0){
        double phiyy_0m0 = qnnn.f_0m0_linear(phi_xx_p[1]);
        double theta_0m0 = interface_Location_With_Second_Order_Derivative(0., d_0m0, phi_000, phi_0m0, phi_xx_p[1][n], phiyy_0m0);
        if (theta_0m0<eps) theta_0m0 = eps; if (theta_0m0>d_0m0) theta_0m0 = d_0m0;
        d_0m0_m0 = d_0m0_p0 = 0;
#ifdef P4_TO_P8
        d_0m0_0m = d_0m0_0p = 0;
#endif
        d_0m0 = theta_0m0;
#ifdef P4_TO_P8
        val_interface_0m0 = bc->interfaceValue(xyz_n[0], xyz_n[1] - theta_0m0, xyz_n[2]);
#else
        val_interface_0m0 = bc->interfaceValue(xyz_n[0], xyz_n[1] - theta_0m0);
#endif
      }
      if( is_interface_0p0){
        double phiyy_0p0 = qnnn.f_0p0_linear(phi_xx_p[1]);
        double theta_0p0 = interface_Location_With_Second_Order_Derivative(0., d_0p0, phi_000, phi_0p0, phi_xx_p[1][n], phiyy_0p0);
        if (theta_0p0<eps) theta_0p0 = eps; if (theta_0p0>d_0p0) theta_0p0 = d_0p0;
        d_0p0_m0 = d_0p0_p0 = 0;
#ifdef P4_TO_P8
        d_0p0_0m = d_0p0_0p = 0;
#endif
        d_0p0 = theta_0p0;
#ifdef P4_TO_P8
        val_interface_0p0 = bc->interfaceValue(xyz_n[0], xyz_n[1] + theta_0p0, xyz_n[2]);
#else
        val_interface_0p0 = bc->interfaceValue(xyz_n[0], xyz_n[1] + theta_0p0);
#endif
      }
#ifdef P4_TO_P8
      if( is_interface_00m){
        double phizz_00m = qnnn.f_00m_linear(phi_xx_p[2]);
        double theta_00m = interface_Location_With_Second_Order_Derivative(0., d_00m, phi_000, phi_00m, phi_xx_p[2][n], phizz_00m);
        if (theta_00m<eps) theta_00m = eps; if (theta_00m>d_00m) theta_00m = d_00m;
        d_00m_m0 = d_00m_p0 = d_00m_0m = d_00m_0p = 0;
        d_00m = theta_00m;
        val_interface_00m = bc->interfaceValue(xyz_n[0], xyz_n[1] , xyz_n[2] - theta_00m);
      }
      if( is_interface_00p){
        double phizz_00p = qnnn.f_00p_linear(phi_xx_p[2]);
        double theta_00p = interface_Location_With_Second_Order_Derivative(0., d_00p, phi_000, phi_00p, phi_xx_p[2][n], phizz_00p);
        if (theta_00p<eps) theta_00p = eps; if (theta_00p>d_00p) theta_00p = d_00p;
        d_00p_m0 = d_00p_p0 = d_00p_0m = d_00p_0p = 0;
        d_00p = theta_00p;
        val_interface_00p = bc->interfaceValue(xyz_n[0], xyz_n[1] , xyz_n[2] + theta_00p);
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

      w_m00 *= wi * mu; w_p00 *= wi * mu;
      w_0m0 *= wj * mu; w_0p0 *= wj * mu;
      w_00m *= wk * mu; w_00p *= wk * mu;

      //---------------------------------------------------------------------
      // diag scaling
      //---------------------------------------------------------------------
      double w_000 = add_p[n] - ( w_m00 + w_p00 + w_0m0 + w_0p0 + w_00m + w_00p );

      double eps_x = is_node_xmWall(p4est, ni) ? 2*EPS : (is_node_xpWall(p4est, ni) ? -2*EPS : 0);
      double eps_y = is_node_ymWall(p4est, ni) ? 2*EPS : (is_node_ypWall(p4est, ni) ? -2*EPS : 0);
      double eps_z = is_node_zmWall(p4est, ni) ? 2*EPS : (is_node_zpWall(p4est, ni) ? -2*EPS : 0);

      //---------------------------------------------------------------------
      // add coefficients to the right hand side
      //---------------------------------------------------------------------
      if(is_node_xmWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0], xyz_n[1]+eps_y, xyz_n[2]+eps_z) / d_p00;
      else if(is_interface_m00)     rhs_p[n] -= w_m00 * val_interface_m00;

      if(is_node_xpWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0], xyz_n[1]+eps_y, xyz_n[2]+eps_z) / d_m00;
      else if(is_interface_p00)     rhs_p[n] -= w_p00 * val_interface_p00;

      if(is_node_ymWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0]+eps_x, xyz_n[1], xyz_n[2]+eps_z) / d_0p0;
      else if(is_interface_0m0)     rhs_p[n] -= w_0m0 * val_interface_0m0;

      if(is_node_ypWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0]+eps_x, xyz_n[1], xyz_n[2]+eps_z) / d_0m0;
      else if(is_interface_0p0)     rhs_p[n] -= w_0p0 * val_interface_0p0;

      if(is_node_zmWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0]+eps_x, xyz_n[1]+eps_y, xyz_n[2]) / d_00p;
      else if(is_interface_00m)     rhs_p[n] -= w_00m * val_interface_00m;

      if(is_node_zpWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0]+eps_x, xyz_n[1]+eps_y, xyz_n[2]) / d_00m;
      if(is_interface_00p)          rhs_p[n] -= w_00p * val_interface_00p;

      rhs_p[n] /= w_000;
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

      w_m00 *= weight_on_Dxx*mu;
      w_p00 *= weight_on_Dxx*mu;
      w_0m0 *= weight_on_Dyy*mu;
      w_0p0 *= weight_on_Dyy*mu;

      //---------------------------------------------------------------------
      // diag scaling
      //---------------------------------------------------------------------

      double diag = add_p[n]-(w_m00+w_p00+w_0m0+w_0p0);

      double eps_x = is_node_xmWall(p4est, ni) ? 2*EPS : (is_node_xpWall(p4est, ni) ? -2*EPS : 0);
      double eps_y = is_node_ymWall(p4est, ni) ? 2*EPS : (is_node_ypWall(p4est, ni) ? -2*EPS : 0);

      if(is_node_xmWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0], xyz_n[1]+eps_y) / d_p00;
      else if(is_interface_m00)     rhs_p[n] -= w_m00*val_interface_m00;

      if(is_node_xpWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0], xyz_n[1]+eps_y) / d_m00;
      else if(is_interface_p00)     rhs_p[n] -= w_p00*val_interface_p00;

      if(is_node_ymWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0]+eps_x, xyz_n[1]) / d_0p0;
      else if(is_interface_0m0)     rhs_p[n] -= w_0m0*val_interface_0m0;

      if(is_node_ypWall(p4est, ni)) rhs_p[n] += 2.*mu*bc->wallValue(xyz_n[0]+eps_x, xyz_n[1]) / d_0m0;
      else if(is_interface_0p0)     rhs_p[n] -= w_0p0*val_interface_0p0;

      rhs_p[n] /= diag;
#endif
      continue;
    }
  }

  if (matrix_has_nullspace && fixed_value_idx_l >= 0){
    rhs_p[fixed_value_idx_l] = 0;
  }


  // restore the pointers
  ierr = VecRestoreArray(phi,    &phi_p   ); CHKERRXX(ierr);
  for(int i=0; i<P4EST_DIM; ++i)
  {
    ierr = VecRestoreArray(phi_xx[i], &phi_xx_p[i]); CHKERRXX(ierr);
  }
  ierr = VecRestoreArray(add,    &add_p   ); CHKERRXX(ierr);
  ierr = VecRestoreArray(rhs,    &phi_p   ); CHKERRXX(ierr);
  if (robin_coef) { ierr = VecGetArray(robin_coef, &robin_coef_p); CHKERRXX(ierr); }

  ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_voronoi_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);
}


#ifdef P4_TO_P8
void my_p4est_poisson_nodes_voronoi_t::set_phi(Vec phi, Vec phi_xx, Vec phi_yy, Vec phi_zz)
#else
void my_p4est_poisson_nodes_voronoi_t::set_phi(Vec phi, Vec phi_xx, Vec phi_yy)
#endif
{
  this->phi = phi;
  is_matrix_computed = false;

#ifdef P4_TO_P8
  if (phi_xx != NULL && phi_yy != NULL && phi_zz != NULL)
#else
  if (phi_xx != NULL && phi_yy != NULL)
#endif
  {
    this->phi_xx[0] = phi_xx;
    this->phi_xx[1] = phi_yy;
#ifdef P4_TO_P8
    this->phi_xx[2] = phi_zz;
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
    ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_xx[0]); CHKERRXX(ierr);
    ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_xx[1]); CHKERRXX(ierr);
#ifdef P4_TO_P8
    ierr = VecCreateGhostNodes(p4est, nodes, &this->phi_xx[2]); CHKERRXX(ierr);
#endif

#ifdef P4_TO_P8
    ngbd->second_derivatives_central(this->phi, this->phi_xx[0], this->phi_xx[1], this->phi_xx[2]);
#else
    ngbd->second_derivatives_central(this->phi, this->phi_xx[0], this->phi_xx[1]);
#endif
    is_phi_dd_owned = true;
  }

  // set the interpolating function parameters
  phi_interp.set_input(this->phi, linear);
}


void my_p4est_poisson_nodes_voronoi_t::shift_to_exact_solution(Vec sol, Vec uex){
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
