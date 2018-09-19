#include "my_p4est_electroporation_solve.h"
#ifdef P4_TO_P8
#include <src/my_p8est_electroporation_solve.h>
#include <src/my_p8est_refine_coarsen.h>
#else
#include <src/my_p4est_electroporation_solve.h>
#include <src/my_p4est_refine_coarsen.h>
#endif

#include <algorithm>

#include <src/matrix.h>
#include <src/my_p4est_solve_lsqr.h>

#include <src/petsc_compatibility.h>
#include <src/math.h>
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

my_p4est_electroporation_solve_t::my_p4est_electroporation_solve_t(const my_p4est_node_neighbors_t *node_neighbors,
                                                                   const my_p4est_cell_neighbors_t *cell_neighbors)
    : ngbd_n(node_neighbors),  ngbd_c(cell_neighbors), myb(node_neighbors->myb),
      p4est(node_neighbors->p4est), ghost(node_neighbors->ghost), nodes(node_neighbors->nodes),
      phi(NULL), rhs(NULL), sol_voro(NULL), vn_voro(NULL),
      voro_global_offset(p4est->mpisize),
      interp_phi(node_neighbors), interp_sol(node_neighbors), interp_dphi_x(node_neighbors),interp_dphi_y(node_neighbors),interp_dphi_z(node_neighbors),
      rhs_m(node_neighbors),
      rhs_p(node_neighbors),
      local_mu(false), local_add(false),
      local_u_jump(false), local_mu_grad_u_jump(false),local_vn(false), local_vnm1(false), local_vnm2(false), local_Sm(false), local_X0(false), local_X1(false),
      mu_m(&mu_constant), mu_p(&mu_constant), add(&add_constant),
      u_jump(&zero), mu_grad_u_jump(&zero), vn(&zero), vnm1(&zero), vnm2(&zero), Smm(&zero), beta_0(NULL), beta_1(NULL),
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


my_p4est_electroporation_solve_t::~my_p4est_electroporation_solve_t()
{
    if(A            != PETSC_NULL) { ierr = MatDestroy(A);                     CHKERRXX(ierr); }
    if(A_null_space != PETSC_NULL) { ierr = MatNullSpaceDestroy(A_null_space); CHKERRXX(ierr); }
    if(ksp          != PETSC_NULL) { ierr = KSPDestroy(ksp);                   CHKERRXX(ierr); }
    if(rhs          != PETSC_NULL) { ierr = VecDestroy(rhs);                   CHKERRXX(ierr); }
    if(local_vn)            { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(vn); }
    if(local_Sm)            { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(Smm); }
    if(local_X0)            { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(X0); }
    if(local_X1)            { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(X1); }
    if(local_vnm1)            { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(vnm1); }
    if(local_vnm2)            { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(vnm2); }
    if(local_mu)             { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_m); delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_p); }
    if(local_add)            { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(add); }
    if(local_u_jump)         { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(u_jump); }
    if(local_mu_grad_u_jump) { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_grad_u_jump); }
    /*if(X0_voro          != PETSC_NULL) { ierr = VecDestroy(X0_voro); CHKERRXX(ierr);}
    if(X1_voro          != PETSC_NULL) { ierr = VecDestroy(X1_voro); CHKERRXX(ierr);}
    if(Sm_voro          != PETSC_NULL) { ierr = VecDestroy(Sm_voro); CHKERRXX(ierr);}
    if(sol_voro          != PETSC_NULL) { ierr = VecDestroy(sol_voro); CHKERRXX(ierr);}
    if(vn_voro          != PETSC_NULL) { ierr = VecDestroy(vn_voro); CHKERRXX(ierr);}*/
}


PetscErrorCode my_p4est_electroporation_solve_t::VecCreateGhostVoronoiRhs()
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

    ierr = VecCreateGhost(p4est->mpicomm, num_local_voro, num_global,
                          ghost_voro.size(), (const PetscInt*)&ghost_voro[0], &rhs); CHKERRQ(ierr);
    ierr = VecSetFromOptions(rhs); CHKERRQ(ierr);

    return ierr;
}
void my_p4est_electroporation_solve_t::set_parameters(int implicit_in, int order_in, double dt_in, int test_in, double SL_in,double tau_ep_in,double tau_res_in,double tau_perm_in, double S0_in, double S1_in, double tn_in)
{
    implicit = implicit_in;
    order = order_in;
    dt = dt_in;
    test = test_in;
    SL = SL_in;
    tau_ep = tau_ep_in;
    tau_res = tau_res_in;
    tau_perm = tau_perm_in;
    S0 = S0_in;
    S1 = S1_in;
    t = tn_in;
}

void my_p4est_electroporation_solve_t::set_Sm(Vec Sm_m)
{
    if(local_Sm) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->Smm);
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(Sm_m, linear);
    this->Smm = tmp;
    local_Sm = true;
    is_matrix_computed = false;
}

void my_p4est_electroporation_solve_t::set_X0(Vec X0)
{
    if(local_X0) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->X0);
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(X0, linear);
    this->X0 = tmp;
    local_X0 = true;
}

void my_p4est_electroporation_solve_t::set_X1(Vec X1)
{
    if(local_X1) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->X1);
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(X1, linear);
    this->X1 = tmp;
    local_X1 = true;
}


void my_p4est_electroporation_solve_t::set_phi(Vec phi)
{
    this->phi = phi;
    interp_phi.set_input(phi, linear);
}

#ifdef P4_TO_P8
void my_p4est_electroporation_solve_t::set_grad_phi(Vec grad_phi[3])
#else
void my_p4est_electroporation_solve_t::set_grad_phi(Vec grad_phi[2])
#endif
{
    interp_dphi_x.set_input(grad_phi[0], linear);
    interp_dphi_y.set_input(grad_phi[1], linear);
#ifdef P4_TO_P8
    interp_dphi_z.set_input(grad_phi[2], linear);
#endif
}


void my_p4est_electroporation_solve_t::set_rhs(Vec rhs_m, Vec rhs_p)
{
    this->rhs_m.set_input(rhs_m, linear);
    this->rhs_p.set_input(rhs_p, linear);
}

void my_p4est_electroporation_solve_t::set_vn(Vec vn)
{
    if(local_vn) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->vn);
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(vn, linear);
    this->vn = tmp;
    local_vn = true;
}

void my_p4est_electroporation_solve_t::set_vnm1(Vec vnm1)
{
    if(local_vnm1) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->vnm1);
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(vnm1, linear);
    this->vnm1 = tmp;
    local_vnm1 = true;
}

void my_p4est_electroporation_solve_t::set_vnm2(Vec vnm2)
{
    if(local_vnm2) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->vnm2);
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(vnm2, linear);
    this->vnm2 = tmp;
    local_vnm2 = true;
}

void my_p4est_electroporation_solve_t::set_diagonal(double add)
{
    if(local_add) { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->add); local_add = false; }
    add_constant.set(add);
    this->add = &add_constant;
}

void my_p4est_electroporation_solve_t::set_diagonal(Vec add)
{
    if(local_add) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->add);
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(add, linear);
    this->add = tmp;
    local_add = true;
}


#ifdef P4_TO_P8
void my_p4est_electroporation_solve_t::set_bc(BoundaryConditions3D& bc)
#else
void my_p4est_electroporation_solve_t::set_bc(BoundaryConditions2D& bc)
#endif
{
    this->bc = &bc;
    is_matrix_computed = false;
}


void my_p4est_electroporation_solve_t::set_beta0(CF_1& beta_0_in)
{
    this->beta_0 = &beta_0_in;
}

void my_p4est_electroporation_solve_t::set_beta1(CF_1& beta_1_in)
{
    this->beta_1 = &beta_1_in;
}

void my_p4est_electroporation_solve_t::set_mu(double mu)
{
    if(local_mu) { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_m); delete dynamic_cast<my_p4est_interpolation_nodes_t*>(mu_p); local_mu = false; }
    mu_constant.set(mu);
    mu_m = &mu_constant;
    mu_p = &mu_constant;
}


void my_p4est_electroporation_solve_t::set_mu(Vec mu_m, Vec mu_p)
{
    if(local_mu) { delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->mu_m); delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->mu_p); }
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(mu_m, linear);
    this->mu_m = tmp;

    tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(mu_p, linear);
    this->mu_p = tmp;
    local_mu = true;
}


void my_p4est_electroporation_solve_t::set_u_jump(Vec u_jump)
{
    if(local_u_jump) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->u_jump);
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(u_jump, linear);
    this->u_jump = tmp;
    local_u_jump = true;
}

void my_p4est_electroporation_solve_t::set_mu_grad_u_jump(Vec mu_grad_u_jump)
{
    if(local_mu_grad_u_jump) delete dynamic_cast<my_p4est_interpolation_nodes_t*>(this->mu_grad_u_jump);
    my_p4est_interpolation_nodes_t *tmp = new my_p4est_interpolation_nodes_t(ngbd_n);
    tmp->set_input(mu_grad_u_jump, linear);
    this->mu_grad_u_jump = tmp;
    local_mu_grad_u_jump = true;
}


void my_p4est_electroporation_solve_t::solve(Vec solution, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type)
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
        ierr = PetscPrintf(p4est->mpicomm, "Computing voronoi points ...\n"); CHKERRXX(ierr);
        compute_voronoi_points();
        ierr = PetscPrintf(p4est->mpicomm, "Done computing voronoi points.\n"); CHKERRXX(ierr);
    }

    /*
   * Here we set the matrix, ksp, and pc. If the matrix is not changed during
   * successive solves, we will reuse the same preconditioner, otherwise we
   * have to recompute the preconditioner
   */
    if(!is_matrix_computed)
    {
        matrix_has_nullspace = true;

        ierr = PetscPrintf(p4est->mpicomm, "Assembling linear system ...\n"); CHKERRXX(ierr);
        setup_linear_system();
        ierr = PetscPrintf(p4est->mpicomm, "Done assembling linear system.\n"); CHKERRXX(ierr);

        is_matrix_computed = true;   //PAM: bc electroporation affects the A matrix



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

    ierr = PetscPrintf(p4est->mpicomm, "Solving linear system ...\n"); CHKERRXX(ierr);
    ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_KSPSolve, ksp, rhs, sol_voro, 0); CHKERRXX(ierr);
    ierr = KSPSolve(ksp, rhs, sol_voro); CHKERRXX(ierr);
    ierr = PetscLogEventEnd  (log_PoissonSolverNodeBasedJump_KSPSolve, ksp, rhs, sol_voro, 0); CHKERRXX(ierr);
    ierr = PetscPrintf(p4est->mpicomm, "Done solving linear system.\n"); CHKERRXX(ierr);

    /* update ghosts */
    ierr = VecGhostUpdateBegin(sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (sol_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    /* interpolate the solution back onto the original mesh */
    interpolate_solution_from_voronoi_to_tree(solution);
    ierr = VecDestroy(sol_voro); CHKERRXX(ierr);
    ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_solve, A, rhs, ksp, 0); CHKERRXX(ierr);
}


void my_p4est_electroporation_solve_t::compute_voronoi_points()
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
void my_p4est_electroporation_solve_t::compute_voronoi_cell(unsigned int n, Voronoi3D &voro) const
#else
void my_p4est_electroporation_solve_t::compute_voronoi_cell(unsigned int n, Voronoi2D &voro) const
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
    p4est_locidx_t quad_idx;
#ifdef CASL_THROWS
    if(rank_found==-1)
        throw std::invalid_argument("[CASL_ERROR]: my_p4est_electroporation_solve_t->compute_voronoi_mesh: found remote quadrant.");
#endif
    if(rank_found==p4est->mpirank) quad_idx = quad.p.piggy3.local_num + tree->quadrants_offset;
    else                           quad_idx = quad.p.piggy3.local_num + p4est->local_num_quadrants;

    double qh[P4EST_DIM];
    dxyz_quad(p4est, &quad, qh);

    double qx = quad_x_fr_q(quad_idx, tree_idx, p4est, ghost);
    double qy = quad_y_fr_q(quad_idx, tree_idx, p4est, ghost);
#ifdef P4_TO_P8
    double qz = quad_z_fr_q(quad_idx, tree_idx, p4est, ghost);
#endif

#ifdef P4_TO_P8
    voro.set_Center_Point(n, pc);
#else
    voro.set_Center_Point(pc);
#endif

    std::vector<p4est_locidx_t> ngbd_quads;

    /* if exactly on a grid node */
    if( (fabs(xyz[0]-(qx-qh[0]/2))<EPS || fabs(xyz[0]-(qx+qh[0]/2))<EPS) &&
            (fabs(xyz[1]-(qy-qh[1]/2))<EPS || fabs(xyz[1]-(qy+qh[1]/2))<EPS)
        #ifdef P4_TO_P8
            && (fabs(xyz[2]-(qz-qh[2]/2))<EPS || fabs(xyz[2]-(qz+qh[2]/2))<EPS)
        #endif
            )
    {
#ifdef P4_TO_P8
        int dir = (fabs(xyz[0]-(qx-qh[0]/2))<EPS ?
                    (fabs(xyz[1]-(qy-qh[1]/2))<EPS ?
                    (fabs(xyz[2]-(qz-qh[2]/2))<EPS ? dir::v_mmm : dir::v_mmp)
            : (fabs(xyz[2]-(qz-qh[2]/2))<EPS ? dir::v_mpm : dir::v_mpp) )
            : (fabs(xyz[1]-(qy-qh[1]/2))<EPS ?
            (fabs(xyz[2]-(qz-qh[2]/2))<EPS ? dir::v_pmm : dir::v_pmp)
            : (fabs(xyz[2]-(qz-qh[2]/2))<EPS ? dir::v_ppm : dir::v_ppp) ) );
#else
        int dir = (fabs(xyz[0]-(qx-qh[0]/2))<EPS ?
                    (fabs(xyz[1]-(qy-qh[1]/2))<EPS ? dir::v_mmm : dir::v_mpm)
            : (fabs(xyz[1]-(qy-qh[1]/2))<EPS ? dir::v_pmm : dir::v_ppm) );
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
#ifndef P4_TO_P8
    if(is_quad_xmWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_m00, pc.x-MAX(EPS, 2*(pc.x-xyz_min[0])), pc.y );
    if(is_quad_xpWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_p00, pc.x+MAX(EPS, 2*(xyz_max[0]-pc.x)), pc.y );
    if(is_quad_ymWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_0m0, pc.x, pc.y-MAX(EPS, 2*(pc.y-xyz_min[1])));
    if(is_quad_ypWall(p4est, quad.p.piggy3.which_tree, &quad)) voro.push(WALL_0p0, pc.x, pc.y+MAX(EPS, 2*(xyz_max[1]-pc.y)));
#endif

    /* finally, construct the partition */
#ifdef P4_TO_P8
    const double xyz_min1 [] = {xyz_min[0], xyz_min[1], xyz_min[2]};
    const double xyz_max1 [] = {xyz_max[0], xyz_max[1], xyz_max[2]};
//    double xyz_min [] = {xmin, ymin, zmin};
//    double xyz_max [] = {xmax, ymax, zmax};
    bool periodic[] = {false, false, false};
    voro.construct_Partition(xyz_min1, xyz_max1, periodic);
#else
    voro.construct_Partition();
#endif

    ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_compute_voronoi_cell, 0, 0, 0, 0); CHKERRXX(ierr);
}

void my_p4est_electroporation_solve_t::compute_electroporation()
{
    PetscPrintf(p4est->mpicomm, "Begin computing electroporation on the Voronoi mesh.\n");
    VecDuplicate(sol_voro, &vn_voro);
    double *sol_voro_p;
    ierr = VecGetArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);

    VecDuplicate(sol_voro, &X0_voro);
    VecDuplicate(sol_voro, &X1_voro);
    VecDuplicate(sol_voro, &Sm_voro);

    for(unsigned int n=0; n<num_local_voro; ++n)
    {
#ifdef P4_TO_P8
        Voronoi3D voro;
#else
        Voronoi2D voro;
#endif

        compute_voronoi_cell(n, voro);

#ifdef P4_TO_P8
        Point3 pc;
        Point3 pair;
#else
        Point2 pc;
        Point2 pair;
#endif
        pc = voro.get_Center_Point();
#ifdef P4_TO_P8
        double phi_n = interp_phi(pc.x, pc.y, pc.z);
        double dphix = interp_dphi_x(pc.x,pc.y,pc.z);
        double dphiy = interp_dphi_y(pc.x,pc.y,pc.z);
        double dphiz = interp_dphi_z(pc.x,pc.y,pc.z);
        Point3 dphi;
        dphi.x = dphix;
        dphi.y = dphiy;
        dphi.z = dphiz;
#else
        double phi_n = interp_phi(pc.x, pc.y);
        double dphix = interp_dphi_x(pc.x,pc.y);
        double dphiy = interp_dphi_y(pc.x,pc.y);
        Point2 dphi;
        dphi.x = dphix;
        dphi.y = dphiy;
#endif
        if(fabs(phi_n)>0.3*diag_min || fabs(phi_n)<EPS)
            continue;

        pair = pc-dphi*phi_n*2;
        int pair_id = find_voro_cell_at(pair);

        pair_id -= voro_global_offset[p4est->mpirank];

        if(pair_id<0)
            continue;


#ifdef P4_TO_P8
        Point3 pl = voro_points[pair_id];
        double phi_l = interp_phi(pl.x, pl.y, pl.z);
#else
        Point2 pl = voro_points[pair_id];
        double phi_l = interp_phi(pl.x, pl.y);
#endif


        if(phi_n*phi_l<0)
        {
#ifdef P4_TO_P8
            Voronoi3D voro_n1;
#else
            Voronoi2D voro_n1;
#endif
            compute_voronoi_cell(pair_id, voro_n1);
            /* extend from ex/in-terior to in/ex-terior on Voronoi mesh,*/
#ifdef P4_TO_P8
            double sol_l = interp_sol(pl.x,pl.y,pl.z);
            double sol_n = interp_sol(pc.x,pc.y,pc.z);
#else
            double sol_l = interp_sol(pl.x,pl.y);
            double sol_n = interp_sol(pc.x,pc.y);
#endif
            double u0 = extend_over_interface_Voronoi(voro, voro_n1, phi_l, sol_l);
            double u1 = extend_over_interface_Voronoi(voro_n1, voro, phi_n, sol_n);
            /* compute the two approximations to jump around the interface */
            double v0, v1;
            if(phi_n>0)
            {
                v0 = sol_n - u0;
                v1 = u1 - sol_l;

            } else {
                v0 = u0 - sol_n;
                v1 = sol_l - u1;
            }


            /* assign the average jump to both the Voronoi points across the interface */
            double jump_v;

            jump_v = 0.5*(v0 + v1);


            int global_ghost_idx;
            int global_voro_idx;
            if(pair_id>=num_local_voro)
                global_ghost_idx = voro_ghost_local_num[pair_id-num_local_voro] + voro_global_offset[voro_ghost_rank[pair_id-num_local_voro]];
            else
                global_ghost_idx = pair_id + voro_global_offset[p4est->mpirank];

            global_voro_idx = n + voro_global_offset[p4est->mpirank];

#ifdef P4_TO_P8
            double X0_n = (*X0)(pc.x, pc.y, pc.z);
            double X1_n = (*X1)(pc.x, pc.y, pc.z);
#else
            double X0_n = (*X0)(pc.x, pc.y);
            double X1_n = (*X1)(pc.x, pc.y);
#endif
            double X0_tmp = X0_n + dt*(((*beta_0)(jump_v) - X0_n)/tau_ep);
            double X1_voro_n = X1_n + dt*MAX( ((*beta_1)(X0_n)-X1_n)/tau_perm, ((*beta_1)(X0_n)-X1_n)/tau_res );
            double X0_voro_n = X0_tmp;
            double Sm_voro_n = SL + S0*X0_voro_n + S1*X1_voro_n;


            VecSetValues(vn_voro,1,&global_voro_idx,&jump_v,INSERT_VALUES);
            VecSetValues(X0_voro,1,&global_voro_idx,&X0_voro_n,INSERT_VALUES);
            VecSetValues(X1_voro,1,&global_voro_idx,&X1_voro_n,INSERT_VALUES);
            VecSetValues(Sm_voro,1,&global_voro_idx,&Sm_voro_n,INSERT_VALUES);

            VecSetValues(vn_voro,1,&global_ghost_idx,&jump_v,INSERT_VALUES);
            VecSetValues(X0_voro,1,&global_ghost_idx,&X0_voro_n,INSERT_VALUES);
            VecSetValues(X1_voro,1,&global_ghost_idx,&X1_voro_n,INSERT_VALUES);
            VecSetValues(Sm_voro,1,&global_ghost_idx,&Sm_voro_n,INSERT_VALUES);
        }
    }

    VecAssemblyBegin(vn_voro);
    VecAssemblyEnd(vn_voro);

    VecAssemblyBegin(X0_voro);
    VecAssemblyEnd(X0_voro);

    VecAssemblyBegin(X1_voro);
    VecAssemblyEnd(X1_voro);

    VecAssemblyBegin(Sm_voro);
    VecAssemblyEnd(Sm_voro);

    ierr = VecGhostUpdateBegin(vn_voro,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(vn_voro,INSERT_VALUES,SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(X0_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (X0_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(X1_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (X1_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(Sm_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (Sm_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecGhostUpdateBegin(vn_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd  (vn_voro, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

    ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);

    PetscPrintf(p4est->mpicomm, "END computing electroporation on the Voronoi mesh.\n");
}

#ifdef P4_TO_P8
double my_p4est_electroporation_solve_t::extend_over_interface_Voronoi(Voronoi3D voro_n0, Voronoi3D voro_n1, double phi_n1, double sol_n1)
#else
double my_p4est_electroporation_solve_t::extend_over_interface_Voronoi(Voronoi2D voro_n0, Voronoi2D voro_n1, double phi_n1, double sol_n1)
#endif
{
    // extends the solution from voronoi nodes downstream from "n1"  crossing over the interface to the voronoi node "n0"
#ifdef P4_TO_P8
    Point3 pc= voro_n0.get_Center_Point();
    Point3 norm = voro_n1.get_Center_Point() - voro_n0.get_Center_Point();
    double d0 = sqrt(norm.x*norm.x+norm.y*norm.y+norm.z*norm.z);
#else
    Point2 pc= voro_n0.get_Center_Point();
    Point2 norm = voro_n1.get_Center_Point() - voro_n0.get_Center_Point();
    double d0 = sqrt(norm.x*norm.x+norm.y*norm.y);
    if(d0<EPS)
        return sol_n1;
#endif

    double diagg = 5*d0/2;
    norm.x /= d0;
    norm.y /= d0;
#ifdef P4_TO_P8
    norm.z /= d0;
#endif
    double p1 = sol_n1;
    double d1 = 0;
    double d2 = diagg;

    double p2;
#ifdef P4_TO_P8
    Point3 tmp;
    Point3 tmp3;
#else
    Point2 tmp;
    Point2 tmp3;
#endif
    tmp = pc + norm*(d0 + diagg);
#ifdef P4_TO_P8
    tmp.x = std::min(xyz_max[0], std::max(xyz_min[0], tmp.x));
    tmp.y = std::min(xyz_max[1], std::max(xyz_min[1], tmp.y));
    tmp.z = std::min(xyz_max[2], std::max(xyz_min[2], tmp.z));
    p2 = interp_sol(tmp.x,tmp.y,tmp.z);
#else
    tmp.x = std::min(xyz_max[0], std::max(xyz_min[0], tmp.x));
    tmp.y = std::min(xyz_max[1], std::max(xyz_min[1], tmp.y));
    p2 = interp_sol(tmp.x,tmp.y);
#endif

    double d3 = 2*diagg;
    tmp3 = pc + norm*(d0 + 2*diagg);

    tmp3.x = std::min(xyz_max[0], std::max(xyz_min[0], tmp3.x));
    tmp3.y = std::min(xyz_max[1], std::max(xyz_min[1], tmp3.y));
#ifdef P4_TO_P8
    tmp3.z = std::min(xyz_max[2], std::max(xyz_min[2], tmp3.z));
#endif

    double p3;
#ifdef P4_TO_P8
    p3 = interp_sol(tmp3.x,tmp3.y,tmp3.z);
#else
    p3 = interp_sol(tmp3.x,tmp3.y);
#endif

    double dif01 = (p2-p1)/(d2-d1);
    double dif12 = (p3-p2)/(d3-d2);
    double dif012 = (dif12-dif01)/(d3-d1);

    return p1 + (-d0-d1)*dif01 + (-d0-d1)*(-d0-d2)*dif012;
}





#ifdef P4_TO_P8
int my_p4est_electroporation_solve_t::find_voro_cell_at(Point3 p_int)
#else
int my_p4est_electroporation_solve_t::find_voro_cell_at(Point2 p_int)
#endif
{
    double xyz1 [] =
    {
        std::min(xyz_max[0], std::max(xyz_min[0], p_int.x)),
        std::min(xyz_max[1], std::max(xyz_min[1], p_int.y))
    #ifdef P4_TO_P8
        , std::min(xyz_max[2], std::max(xyz_min[2], p_int.z))
    #endif
    };

    p4est_quadrant_t quad;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz1, quad, remote_matches);

    unsigned int index = -1; // this is the index of the voro cell containing the point p_int necessary for interpolation
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
#ifdef P4_TO_P8
        Point3 vp;
#else
        Point2 vp;
#endif
        double tmp_dist_voro = DBL_MAX;
        for(unsigned int m=0; m<grid2voro[node].size(); ++m)
        {

            vp = voro_points[grid2voro[node][m]];

            double voros_dist = (p_int-vp).norm_L2();
            if(voros_dist<tmp_dist_voro)
            {
                tmp_dist_voro = voros_dist;
                index = grid2voro[node][m];
            }
        }

        return index + voro_global_offset[p4est->mpirank];

    }
}



#ifdef P4_TO_P8
int my_p4est_electroporation_solve_t::find_tree_node_at(Point3 p_int)
#else
int my_p4est_electroporation_solve_t::find_tree_node_at(Point2 p_int)
#endif
{
    double xyz1 [] =
    {
        std::min(xyz_max[0], std::max(xyz_min[0], p_int.x)),
        std::min(xyz_max[1], std::max(xyz_min[1], p_int.y))
    #ifdef P4_TO_P8
        , std::min(xyz_max[2], std::max(xyz_min[2], p_int.z))
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

        return node;

    }
}





#ifdef P4_TO_P8
double my_p4est_electroporation_solve_t::interpolate_Voronoi_at_point(Point3 p_int, double sign)
#else
double my_p4est_electroporation_solve_t::interpolate_Voronoi_at_point(Point2 p_int, double sign)
#endif
{
    //developing interpolate_Voronoi_at_point(Point3): find *all* the neighborhood voronoi points for tmp (follow the project on tree method below where they find the neighbor voronoi points to a given node),
    // setup a linear system by distances of each of such points to tmp and rhs= solution at each voronoi point. solve the linear system using the solve_lsqr_system function
    //to get the interpolated value of solution on tmp.

    /* gather the neighborhood of voronoi cells */
    std::vector<p4est_locidx_t> ngbd_voro;
#ifdef P4_TO_P8
    std::vector<Point3> ngbd_voro_coords;
#else
    std::vector<Point2> ngbd_voro_coords;
#endif


    double xyz1 [] =
    {
        std::min(xyz_max[0], std::max(xyz_min[0], p_int.x)),
        std::min(xyz_max[1], std::max(xyz_min[1], p_int.y))
    #ifdef P4_TO_P8
        , std::min(xyz_max[2], std::max(xyz_min[2], p_int.z))
    #endif
    };

    p4est_quadrant_t quad;
    std::vector<p4est_quadrant_t> remote_matches;
    int rank_found = ngbd_n->hierarchy->find_smallest_quadrant_containing_point(xyz1, quad, remote_matches);

    unsigned int index = -1; // this is the index of the voro cell containing the point p_int necessary for interpolation
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
        double tmp_dist_voro = DBL_MAX;
        for(unsigned int m=0; m<grid2voro[node].size(); ++m)
        {
#ifdef P4_TO_P8
            Point3 vp = voro_points[grid2voro[node][m]];
#else
            Point2 vp = voro_points[grid2voro[node][m]];
#endif
            double voros_dist = (p_int-vp).norm_L2();
            if(voros_dist<tmp_dist_voro)
            {
                tmp_dist_voro = voros_dist;
                index = grid2voro[node][m];
            }
        }
        if(index==-1)
            return 0;
        PetscPrintf(p4est->mpicomm, "voro_index  %d, node %d, x %g, y %g, z %g\n", index, node, xyz1[0], xyz1[1], xyz1[2]);




#ifdef P4_TO_P8
        Voronoi3D voro_index;
        Voronoi3D voro_tmp;
#else
        Voronoi2D voro_index;
        Voronoi2D voro_tmp;
#endif
        compute_voronoi_cell(index, voro_index);
#ifdef P4_TO_P8
        const std::vector<Voronoi3DPoint> *points;
#else
        const std::vector<Voronoi2DPoint> *points;
#endif
        double scaling = DBL_MAX;


        voro_index.get_Points(points);

        for(unsigned int l=0; l<points->size(); ++l)
            if((*points)[l].n>=0)
            {
                compute_voronoi_cell((*points)[l].n, voro_tmp);
                scaling = MIN(scaling, (voro_index.get_Center_Point() - voro_tmp.get_Center_Point()).norm_L2());
                ngbd_voro.push_back((*points)[l].n);
                ngbd_voro_coords.push_back(voro_tmp.get_Center_Point());
            }

        /*
        int s1 = ngbd_voro.size();
        // PetscPrintf(p4est->mpicomm, "ngbd_voro number of points is %d\n", s1);
        for(unsigned int m=0; m<s1; ++m)
        {
            compute_voronoi_cell(ngbd_voro[m], voro_tmp);
            voro_tmp.get_Points(points);
            for(unsigned int l=0; l<points->size(); ++l)
            {
                if((*points)[l].n>=0)
                {
                    if(std::find(ngbd_voro.begin(), ngbd_voro.end(), (*points)[l].n) == ngbd_voro.end())
                    {
                        ngbd_voro.push_back((*points)[l].n);
                        ngbd_voro_coords.push_back(voro_tmp.get_Center_Point());
                    }
                }
            }
        }


        int s2 = ngbd_voro.size();
        for(unsigned int m=s1; m<s2; ++m)
        {
            compute_voronoi_cell(ngbd_voro[m], voro_tmp);
            voro_tmp.get_Points(points);
            for(unsigned int l=0; l<points->size(); ++l)
            {
                if((*points)[l].n>=0)
                {
                    if(std::find(ngbd_voro.begin(), ngbd_voro.end(), (*points)[l].n) == ngbd_voro.end())
                    {
                        ngbd_voro.push_back((*points)[l].n);
                        ngbd_voro_coords.push_back(voro_tmp.get_Center_Point());
                    }
                }

            }
        }
        */

        //Vec rhs_interface;
        //VecDuplicate(sol_voro, &rhs_interface);
        double *sol_voro_p, *rhs_interface_p;
        VecGetArray(sol_voro, &sol_voro_p);
        //VecGetArray(rhs_interface, &rhs_interface_p);




        //    Mat B;
        //    int num_rows = 0;
        //    for(unsigned int m=0; m<ngbd_voro.size(); m++)
        //    {
        //#ifdef P4_TO_P8
        //        Point3 p_voro = ngbd_voro_coords[m];
        //        double phi_m = interp_phi(p_voro.x, p_voro.y, p_voro.z);
        //#else
        //        Point2 p_voro = ngbd_voro_coords[m];
        //        double phi_m = interp_phi(p_voro.x, p_voro.y);
        //#endif
        //        if(sign==0 || SIGN(phi_m)==sign)
        //        {
        //            num_rows += 1;
        //        }
        //    }
        //    MPI_Allreduce(MPI_IN_PLACE, &num_rows, 1, MPI_INT, MPI_SUM, p4est->mpicomm);
        //    /* set up the matrix */
        //    ierr = MatCreate(p4est->mpicomm, &B); CHKERRXX(ierr);
        //    ierr = MatSetType(B, MATAIJ); CHKERRXX(ierr);
        //    ierr = MatSetSizes(B, PETSC_DECIDE , PETSC_DECIDE, num_rows, 10); CHKERRXX(ierr);
        //    ierr = MatSetFromOptions(B); CHKERRXX(ierr);

        //    /* allocate the matrix */
        //    ierr = MatSeqAIJSetPreallocation(B, 0, (const PetscInt*)&d_nnz[0]); CHKERRXX(ierr);
        //    ierr = MatMPIAIJSetPreallocation(B, 0, (const PetscInt*)&d_nnz[0], 0, (const PetscInt*)&o_nnz[0]); CHKERRXX(ierr);


        //    double rhs_max = 0;
        /* get local number of voronoi points for every processor *//*
    num_local_ngbd_voro = ngbd_voro.size();
    ngbd_voro_global_offset[p4est->mpirank] = num_local_ngbd_voro;
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &ngbd_voro_global_offset[0], 1, MPI_INT, p4est->mpicomm);
    for(int r=1; r<p4est->mpisize; ++r)
    {
        ngbd_voro_global_offset[r] += ngbd_voro_global_offset[r-1];
    }

    ngbd_voro_global_offset.insert(ngbd_voro_global_offset.begin(), 0);*/

        matrix_t A;
#ifdef P4_TO_P8
        A.resize(0, 10);
        std::vector<double> nb_x;
        std::vector<double> nb_y;
        std::vector<double> nb_z;
#else
        A.resize(0, 6);
        std::vector<double> nb_x;
        std::vector<double> nb_y;
#endif
        std::vector<double> p;



        double min_w = 1e-6;
        double inv_max_w = 1e-6;

        for(unsigned int m=0; m<ngbd_voro.size(); m++)
        {
#ifdef P4_TO_P8
            Point3 p_voro = ngbd_voro_coords[m];
            double phi_m = interp_phi(p_voro.x, p_voro.y, p_voro.z);
#else
            Point2 p_voro = ngbd_voro_coords[m];
            double phi_m = interp_phi(p_voro.x, p_voro.y);
#endif
            if(sign==0 || SIGN(phi_m)==sign)
            {
                //PetscInt global_m_idx = ngbd_voro[m]+ngbd_voro_global_offset[p4est->mpirank];
#ifdef P4_TO_P8
                double xt = (p_int.x-p_voro.x)/scaling;
                double yt = (p_int.y-p_voro.y)/scaling;
                double zt = (p_int.z-p_voro.z)/scaling;
                double w = MAX(min_w, 1./MAX(inv_max_w, sqrt(xt*xt+yt*yt+zt*zt)));
#else
                double xt = (p_int.x-p_voro.x)/scaling;
                double yt = (p_int.y-p_voro.y)/scaling;
                double w = MAX(min_w, 1./MAX(inv_max_w, sqrt(xt*xt+yt*yt)));
#endif
#ifdef P4_TO_P8
                //            ierr = MatSetValue(B, global_m_idx, 0, 1     * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 1, xt    * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 2, yt    * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 3, zt    * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 4, xt*x  * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 5, xt*yt * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 6, xt*zt * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 7, yt*yt * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 8, yt*zt * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 9, zt*zt * w, INSERT_VALUES); CHKERRXX(ierr);

                A.set_value(p.size(),0, 1     * w);
                A.set_value(p.size(),1, xt    * w);
                A.set_value(p.size(),2, yt    * w);
                A.set_value(p.size(),3, zt    * w);
                A.set_value(p.size(),4, xt*xt * w);
                A.set_value(p.size(),5, xt*yt * w);
                A.set_value(p.size(),6, xt*zt * w);
                A.set_value(p.size(),7, yt*yt * w);
                A.set_value(p.size(),8, yt*zt * w);
                A.set_value(p.size(),9, zt*zt * w);
#else
                //            ierr = MatSetValue(B, global_m_idx, 0, 1     * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 1, xt    * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 2, yt    * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 3, xt*xt * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 4, xt*yt * w, INSERT_VALUES); CHKERRXX(ierr);
                //            ierr = MatSetValue(B, global_m_idx, 5, yt*yt * w, INSERT_VALUES); CHKERRXX(ierr);

                A.set_value(p.size(),0, 1     * w);
                A.set_value(p.size(),1, xt    * w);
                A.set_value(p.size(),2, yt    * w);
                A.set_value(p.size(),3, xt*xt * w);
                A.set_value(p.size(),4, xt*yt * w);
                A.set_value(p.size(),5, yt*yt * w);
#endif
                p.push_back(sol_voro_p[ngbd_voro[m]]*w);


                //                rhs_interface_p[ngbd_voro[m]] = sol_voro_p[ngbd_voro[m]]*w;
                //                rhs_max = MAX(ABS(sol_voro_p[ngbd_voro[m]]*w), rhs_max);

                if(std::find(nb_x.begin(), nb_x.end(), p_voro.x) == nb_x.end())   nb_x.push_back(p_voro.x);
                if(std::find(nb_y.begin(), nb_y.end(), p_voro.y) == nb_y.end())   nb_y.push_back(p_voro.y);
#ifdef P4_TO_P8
                if(std::find(nb_z.begin(), nb_z.end(), p_voro.z) == nb_z.end())   nb_z.push_back(p_voro.z);
#endif
            }
        }
        VecRestoreArray(sol_voro, &sol_voro_p);
        //    VecRestoreArray(rhs_interface, &rhs_interface_p);
        //    /* assemble the matrix */
        //    ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRXX(ierr);
        //    ierr = MatAssemblyEnd  (B, MAT_FINAL_ASSEMBLY);   CHKERRXX(ierr);
        //    MPI_Allreduce(MPI_IN_PLACE, &rhs_max, 1, MPI_DOUBLE, MPI_MAX, p4est->mpicomm);
        //    MatScale(B, rhs_max);


        //    Vec soln;
        //    // solve here...
        //    ierr = VecCreateSeq(PETSC_COMM_SELF, 1, &soln); CHKERRQ(ierr);

        //    ierr = KSPSetType(ksp, KSPLSQR); CHKERRQ(ierr);
        //    ierr = KSPSetOperators(ksp, B, B, SAME_PRECONDITIONER); CHKERRQ(ierr);
        //    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
        //    ierr = KSPSolve(ksp, rhs, soln); CHKERRQ(ierr);

        //    /* update ghosts */
        //    ierr = VecGhostUpdateBegin(soln, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
        //    ierr = VecGhostUpdateEnd  (soln, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);

        //    VecView(soln, PETSC_VIEWER_STDOUT_WORLD);

        //    PetscInt num_iters;
        //    ierr = KSPGetIterationNumber(ksp, &num_iters); CHKERRQ(ierr);

        //    PetscReal rnorm;
        //    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);

        //    KSPConvergedReason reason;
        //    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);

        //    std::cout << "KSPGetIterationNumber " << num_iters << '\n';
        //    std::cout << "KSPGetResidualNorm " << rnorm << '\n';
        //    std::cout << "KSPConvergedReason " << reason << '\n';

        //    if (B != NULL)
        //        ierr = MatDestroy(B); CHKERRXX(ierr);

        //    return 4;
        A.scale_by_maxabs(p);

#ifdef P4_TO_P8
        double interpolated_value = solve_lsqr_system(A, p, nb_x.size(), nb_y.size(), nb_z.size());
#else
        double interpolated_value = solve_lsqr_system(A, p, nb_x.size(), nb_y.size());
#endif
        PetscPrintf(p4est->mpicomm, "matrix A rows: %d, columns: %d value is %g\n", A.num_rows(), A.num_cols(), interpolated_value);
        return interpolated_value;
    }
}


void my_p4est_electroporation_solve_t::interpolate_electroporation_to_tree(Vec X0_tree, Vec X1_tree, Vec Sm_tree, Vec vn_tree)
{
    PetscPrintf(p4est->mpicomm, "Begin interpolating electroporation variables onto tree mesh. \n");
    Vec l0, l;
    ierr = VecGhostGetLocalForm(sol_voro, &l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(X0_voro, &l0); CHKERRXX(ierr);
    ierr = VecCopy(l0, l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(sol_voro, &l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X0_voro, &l0); CHKERRXX(ierr);
    interpolate_solution_from_voronoi_to_tree(X0_tree);

    ierr = VecGhostGetLocalForm(sol_voro, &l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(X1_voro, &l0); CHKERRXX(ierr);
    ierr = VecCopy(l0, l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(sol_voro, &l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(X1_voro, &l0); CHKERRXX(ierr);
    interpolate_solution_from_voronoi_to_tree(X1_tree);

    ierr = VecGhostGetLocalForm(sol_voro, &l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(Sm_voro, &l0); CHKERRXX(ierr);
    ierr = VecCopy(l0, l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(sol_voro, &l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(Sm_voro, &l0); CHKERRXX(ierr);
    interpolate_solution_from_voronoi_to_tree(Sm_tree);

    ierr = VecGhostGetLocalForm(sol_voro, &l); CHKERRXX(ierr);
    ierr = VecGhostGetLocalForm(vn_voro, &l0); CHKERRXX(ierr);
    ierr = VecCopy(l0, l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(sol_voro, &l); CHKERRXX(ierr);
    ierr = VecGhostRestoreLocalForm(vn_voro, &l0); CHKERRXX(ierr);
    interpolate_solution_from_voronoi_to_tree(vn_tree);


    PetscPrintf(p4est->mpicomm, "End interpolating electroporation variables onto tree mesh. \n");
}


void my_p4est_electroporation_solve_t::setup_linear_system()
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
        Point3 pc = voro_points[n];
#else
        Point2 pc = voro_points[n];
#endif
        if( (ABS(pc.x-xyz_min[0])<EPS || ABS(pc.x-xyz_max[0])<EPS ||
             ABS(pc.y-xyz_min[1])<EPS || ABS(pc.y-xyz_max[1])<EPS
     #ifdef P4_TO_P8
             || ABS(pc.z-xyz_min[2])<EPS || ABS(pc.z-xyz_max[2])<EPS
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

        double sigma_n;
        if(phi_n<0)
        {
#ifdef P4_TO_P8
            rhs_p[n] = this->rhs_m(pc.x, pc.y, pc.z);
            sigma_n     = (*mu_m)(pc.x, pc.y, pc.z);
#else
            rhs_p[n] = this->rhs_m(pc.x, pc.y);
            sigma_n     = (*mu_m)(pc.x, pc.y);
#endif
        }
        else
        {
#ifdef P4_TO_P8
            rhs_p[n] = this->rhs_p(pc.x, pc.y, pc.z);
            sigma_n     = (*mu_p)(pc.x, pc.y, pc.z);
#else
            rhs_p[n] = this->rhs_p(pc.x, pc.y);
            sigma_n     = (*mu_p)(pc.x, pc.y);
#endif
        }


#ifndef P4_TO_P8
        voro.compute_volume();
#endif
        double volume = voro.get_volume();

        rhs_p[n] *= volume;
#ifdef P4_TO_P8
        double add_n = (*add)(pc.x, pc.y, pc.z);
#else
        double add_n = (*add)(pc.x, pc.y);
#endif
        if(add_n>EPS) matrix_has_nullspace = false;

        mat_entry_t ent; ent.n = global_n_idx; ent.val = volume*add_n;
        matrix_entries[n].push_back(ent);

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
                /* regular point */
#ifdef P4_TO_P8
                Point3 pl = (*points)[l].p;
                double phi_l = interp_phi(pl.x, pl.y, pl.z);
#else
                Point2 pl = (*points)[l].p;
                double phi_l = interp_phi(pl.x, pl.y);
#endif
                double d = (pc - pl).norm_L2();

                double sigma_l;

#ifdef P4_TO_P8
                if(phi_l<0) sigma_l = (*mu_m)(pl.x, pl.y, pl.z);
                else        sigma_l = (*mu_p)(pl.x, pl.y, pl.z);
#else
                if(phi_l<0) sigma_l = (*mu_m)(pl.x, pl.y);
                else        sigma_l = (*mu_p)(pl.x, pl.y);
#endif

                PetscInt global_l_idx;
                if((unsigned int)(*points)[l].n<num_local_voro)
                {
                    global_l_idx = (*points)[l].n + voro_global_offset[p4est->mpirank];
                    d_nnz[n]++;
                }
                else
                {
                    global_l_idx = voro_ghost_local_num[(*points)[l].n-num_local_voro] + voro_global_offset[voro_ghost_rank[(*points)[l].n-num_local_voro]];
                    o_nnz[n]++;
                }

#ifdef P4_TO_P8
                double Smn = 0.5*((*Smm)(pc.x, pc.y, pc.z) + (*Smm)(pl.x, pl.y, pl.z));
#else
                double Smn = 0.5*((*Smm)(pc.x, pc.y) + (*Smm)(pl.x, pl.y));
#endif

                double sigma_harmonic = sigma_n*sigma_l/(sigma_n + sigma_l+2*sigma_n*sigma_l*dt/(Cm+dt*Smn)/d);
                if(phi_n*phi_l<0)
                {
                    mat_entry_t ent; ent.n = global_l_idx; ent.val = -2*s*sigma_harmonic/d;
                    matrix_entries[n][0].val += 2*s*sigma_harmonic/d;
                    matrix_entries[n].push_back(ent);
                } else {
                    mat_entry_t ent; ent.n = global_l_idx; ent.val = -s*sigma_n/d;
                    matrix_entries[n][0].val += s*sigma_n/d;
                    matrix_entries[n].push_back(ent);
                }


                double sigma_disc;
                if(phi_n*phi_l<0)
                {
#ifdef P4_TO_P8
                    Point3 p_ln = (pc+pl)/2;
#else
                    Point2 p_ln = (pc+pl)/2;
#endif

#ifdef P4_TO_P8
                    rhs_p[n] += s*sigma_harmonic* SIGN(phi_n) * (*u_jump)(p_ln.x, p_ln.y, p_ln.z)*Cm/((Cm+dt*Smn)*d/2);
#else
                    rhs_p[n] += s*sigma_harmonic* SIGN(phi_n) * (*u_jump)(p_ln.x, p_ln.y)*Cm/((Cm+dt*Smn)*d/2);
#endif


                    //                    if(implicit==0)
                    //                    {
                    //                        switch(order)
                    //                        {
                    //                        case 1:
                    //                            /* ([u]_n+1 - [u]_n)/dt + Sm_n [u]_n = d u_n+1/d n */
                    //                            sigma_disc = 2*sigma_n*sigma_l/(sigma_n + sigma_l + sigma_n*sigma_l/(d/2) * dt/Cm);
                    //                            break;
                    //                        case 2:
                    //                            /* (1.5*[u]_n+1 - 2*[u]_n + .5*[u]_n-1)/dt + Sm_n [u]_n = d u_n+1/d n */
                    //                            sigma_disc = 2*sigma_n*sigma_l/(sigma_n + sigma_l + 2./3. * sigma_n*sigma_l/(d/2) * dt/Cm);
                    //                            break;
                    //                        default:
                    //                            throw std::invalid_argument("Unknown order ... choose 1 or 2 when explicit.");
                    //                        }
                    //                    }
                    //                    else
                    //                    {
                    //                        switch(order)
                    //                        {
                    //                        case 1:
                    //                            // ([u]_n+1 - [u]_n)/dt + Sm_n [u]_n+1 = d u_n+1/d n
                    //                            sigma_disc = 2*sigma_n*sigma_l/(sigma_n + sigma_l + sigma_n*sigma_l/(d/2) * dt/(Cm+dt*Smn));
                    //                            break;
                    //                        case 2:
                    //                            // (1.5*[u]_n+1 - 2*[u]_n + .5*[u]_n-1)/dt + Sm_n [u]_n+1 = d u_n+1/d n
                    //                            sigma_disc = 2*sigma_n*sigma_l/(sigma_n + sigma_l + sigma_n*sigma_l/(d/2) * dt/(1.5*Cm+dt*Smn));
                    //                            break;
                    //                        case 3:
                    //                            // (11/6*[u]_n+1 - 3*[u]_n + 3/2*[u]_n-1 - 1/3*[u]_n-2)/dt + Sm_n [u]_n+1 = d u_n+1/d n
                    //                            sigma_disc = 2*sigma_n*sigma_l/(sigma_n + sigma_l + sigma_n*sigma_l/(d/2) * dt/(11./6.*Cm+dt*Smn));
                    //                            break;
                    //                        default:
                    //                            throw std::invalid_argument("Unknown order ...");
                    //                        }
                    //                    }

                    //                    double sigma_tmp, vi;

                    //                    if(implicit==0)
                    //                    {
                    //                        switch(order)
                    //                        {
                    //                        case 1:
                    //                            /* ([u]_n+1 - [u]_n)/dt + Sm_n [u]_n = d u_n+1/d n */
                    //                            sigma_tmp = sigma_n*sigma_l / (sigma_n + sigma_l + sigma_n*sigma_l/(d/2) * dt/Cm);
                    //                            //                            vi = .5*(vn_n_p[n] + vn_n_p[(*points)[l].n]);
                    //#ifdef P4_TO_P8
                    //                            vi = 0.5*((*vn)(pc.x, pc.y, pc.z) + (*vn)(pl.x, pl.y, pl.z));
                    //#else
                    //                            vi = 0.5*((*vn)(pc.x, pc.y) + (*vn)(pl.x, pl.y));
                    //#endif
                    //                            rhs_p[n] += SIGN(phi_n) * s/(d/2) * sigma_tmp * vi * (1-dt*Smn/Cm);
                    //                            break;
                    //                        case 2:
                    //                            /* (1.5*[u]_n+1 - 2*[u]_n + .5*[u]_n-1)/dt + Sm_n [u]_n = d u_n+1/d n */
                    //                            sigma_tmp = sigma_n*sigma_l / (sigma_n + sigma_l + 2./3. * sigma_n*sigma_l/(d/2) * dt/Cm);
                    //                            //                            vi = .5*2./3.*((2-dt*Smn/Cm)*(vn_n_p[n]+vn_n_p[(*points)[l].n]) - .5*(vnm1_n_p[n]+vnm1_n_p[(*points)[l].n]));
                    //#ifdef P4_TO_P8
                    //                            vi = .5*2./3.*((2-dt*Smn/Cm)*((*vn)(pc.x, pc.y, pc.z)+(*vn)(pl.x, pl.y, pl.z)) - .5*((*vnm1)(pc.x, pc.y, pc.z)+(*vnm1)(pl.x, pl.y, pl.z)));
                    //#else
                    //                            vi = .5*2./3.*((2-dt*Smn/Cm)*((*vn)(pc.x, pc.y)+(*vn)(pl.x, pl.y)) - .5*((*vnm1)(pc.x, pc.y)+(*vnm1)(pl.x, pl.y)));
                    //#endif
                    //                            rhs_p[n] += SIGN(phi_n) * s/(d/2) * sigma_tmp * vi;

                    //                            break;
                    //                        default:
                    //                            throw std::invalid_argument("Unknown order ... choose 1 or 2 when explicit.");
                    //                        }
                    //                    }
                    //                    else
                    //                    {
                    //                        switch(order)
                    //                        {
                    //                        case 1:
                    //                            /* ([u]_n+1 - [u]_n)/dt + Sm_n [u]_n+1 = d u_n+1/d n */
                    //                            sigma_tmp = sigma_n*sigma_l / (sigma_n + sigma_l + sigma_n*sigma_l/(d/2) * dt/(Cm+dt*Smn));

                    //                            //vi = .5*(vn_n_p[n] + vn_n_p[(*points)[l].n]);
                    //#ifdef P4_TO_P8
                    //                            vi = 0.5*((*vn)(pc.x, pc.y, pc.z) + (*vn)(pl.x, pl.y, pl.z));
                    //#else
                    //                            vi = 0.5*((*vn)(pc.x, pc.y) + (*vn)(pl.x, pl.y));
                    //#endif
                    //                            rhs_p[n] += SIGN(phi_n) * s/(d/2) * sigma_tmp * Cm*vi / (Cm+dt*Smn);
                    //                            break;
                    //                        case 2:
                    //                            /* (1.5*[u]_n+1 - 2*[u]_n + .5*[u]_n-1)/dt + Sm_n [u]_n+1 = d u_n+1/d n */
                    //                            sigma_tmp = sigma_n*sigma_l / (sigma_n + sigma_l + sigma_n*sigma_l/(d/2) * dt/(1.5*Cm+dt*Smn));
                    //                            //vi = .5*(2*vn_n_p[n] - .5*vnm1_n_p[n] + 2*vn_n_p[(*points)[l].n] - .5*vnm1_n_p[(*points)[l].n]);
                    //#ifdef P4_TO_P8
                    //                            vi = .5*(2*(*vn)(pc.x, pc.y, pc.z) - .5*(*vnm1)(pc.x, pc.y, pc.z) + 2*(*vn)(pl.x, pl.y, pl.z) - .5*(*vnm1)(pl.x, pl.y, pl.z));
                    //#else
                    //                            vi = .5*(2*(*vn)(pc.x, pc.y) - .5*(*vnm1)(pc.x, pc.y) + 2*(*vn)(pl.x, pl.y) - .5*(*vnm1)(pl.x, pl.y));
                    //#endif
                    //                            rhs_p[n] += SIGN(phi_n) * s/(d/2) * sigma_tmp * Cm*vi / (1.5*Cm+dt*Smn);
                    //                            break;
                    //                        case 3:
                    //                            /* (11/6*[u]_n+1 - 3*[u]_n + 3/2*[u]_n-1 - 1/3*[u]_n-2)/dt + Sm_n [u]_n+1 = d u_n+1/d n */
                    //                            sigma_tmp = sigma_n*sigma_l / (sigma_n + sigma_l + sigma_n*sigma_l/(d/2) * dt/(11./6.*Cm+dt*Smn));
                    //                            //vi = .5*(3*vn_n_p[n] - 3./2.*vnm1_n_p[n] + 1./3.*vnm2_n_p[n] + 3*vn_n_p[(*points)[l].n] - 3./2.*vnm1_n_p[(*points)[l].n] + 1./3.*vnm2_n_p[(*points)[l].n]);
                    //#ifdef P4_TO_P8
                    //                            vi = .5*(3*(*vn)(pc.x,pc.y,pc.z) - 3./2.*(*vnm1)(pc.x,pc.y,pc.z) + 1./3.*(*vnm2)(pc.x,pc.y,pc.z) + 3*(*vn)(pl.x,pl.y,pl.z) - 3./2.*(*vnm1)(pl.x,pl.y,pl.z) + 1./3.*(*vnm2)(pl.x,pl.y,pl.z));
                    //#else
                    //                            vi = .5*(3*(*vn)(pc.x,pc.y) - 3./2.*(*vnm1)(pc.x,pc.y) + 1./3.*(*vnm2)(pc.x,pc.y) + 3*(*vn)(pl.x,pl.y) - 3./2.*(*vnm1)(pl.x,pl.y) + 1./3.*(*vnm2)(pl.x,pl.y));
                    //#endif
                    //                            rhs_p[n] += SIGN(phi_n) * s/(d/2) * sigma_tmp * Cm*vi / (11./6.*Cm+dt*Smn);

                    //                            break;
                    //                        default:
                    //                            throw std::invalid_argument("Unknown order ...");
                    //                        }
                    //                    }

                    //                }
                    //                else
                    //                {
                    //                    sigma_disc = sigma_n;
                    //                }

                    //                mat_entry_t ent1; ent1.n = global_n_idx; ent1.val = +s*sigma_disc/d;
                    //                matrix_entries[n].push_back(ent1);
                    //                mat_entry_t ent; ent.n = global_l_idx; ent.val = -s*sigma_disc/d;
                    //                matrix_entries[n].push_back(ent);
                }
            }
            else /* wall with neumann */
            {
                double x_tmp = pc.x;
                double y_tmp = pc.y;

                /* perturb the corners to differentiate between the edges of the domain ... otherwise 1st order only at the corners */
#ifdef P4_TO_P8
                double z_tmp = pc.z;
                if(fabs(pc.x-xyz_min[0])<EPS && ( (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp += 2*EPS;
                if(fabs(pc.x-xyz_max[0])<EPS && ( (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp -= 2*EPS;
                if(fabs(pc.y-xyz_min[1])<EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp += 2*EPS;
                if(fabs(pc.y-xyz_max[1])<EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp -= 2*EPS;
                if(fabs(pc.z-xyz_min[2])<EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp += 2*EPS;
                if(fabs(pc.z-xyz_max[2])<EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp -= 2*EPS;
                rhs_p[n] += s*sigma_n * bc->wallValue(x_tmp, y_tmp, z_tmp);
#else
                if(fabs(pc.x-xyz_min[0])<EPS && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp += 2*EPS;
                if(fabs(pc.x-xyz_max[0])<EPS && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp -= 2*EPS;
                if(fabs(pc.y-xyz_min[1])<EPS && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp += 2*EPS;
                if(fabs(pc.y-xyz_max[1])<EPS && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp -= 2*EPS;
                rhs_p[n] += s*sigma_n * bc->wallValue(x_tmp, y_tmp);
#endif
            }
        }
    }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

    PetscInt num_owned_global = voro_global_offset[p4est->mpisize];
    PetscInt num_owned_local  = (PetscInt) num_local_voro;

    if (A != NULL)
        ierr = MatDestroy(A); CHKERRXX(ierr);

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
        //    ierr = MatNullSpaceRemove(A_null_space, rhs, NULL); CHKERRXX(ierr);
    }

    ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_setup_linear_system, A, 0, 0, 0); CHKERRXX(ierr);
}


void my_p4est_electroporation_solve_t::setup_negative_laplace_rhsvec()
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
        if( (ABS(pc.x-xyz_min[0])<EPS || ABS(pc.x-xyz_max[0])<EPS ||
             ABS(pc.y-xyz_min[1])<EPS || ABS(pc.y-xyz_max[1])<EPS
     #ifdef P4_TO_P8
             || ABS(pc.z-xyz_min[2])<EPS || ABS(pc.z-xyz_max[2])<EPS
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

        double sigma_n;
        if(phi_n<0)
        {
#ifdef P4_TO_P8
            rhs_p[n] = this->rhs_m(pc.x, pc.y, pc.z);
            sigma_n     = (*mu_m)(pc.x, pc.y, pc.z);
#else
            rhs_p[n] = this->rhs_m(pc.x, pc.y);
            sigma_n     = (*mu_m)(pc.x, pc.y);
#endif
        }
        else
        {
#ifdef P4_TO_P8
            rhs_p[n] = this->rhs_p(pc.x, pc.y, pc.z);
            sigma_n     = (*mu_p)(pc.x, pc.y, pc.z);
#else
            rhs_p[n] = this->rhs_p(pc.x, pc.y);
            sigma_n     = (*mu_p)(pc.x, pc.y);
#endif
        }


#ifndef P4_TO_P8
        voro.compute_volume();
#endif
        double volume = voro.get_volume();

        rhs_p[n] *= volume;

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
                /* regular point */
#ifdef P4_TO_P8
                Point3 pl = (*points)[l].p;
                double phi_l = interp_phi(pl.x, pl.y, pl.z);
#else
                Point2 pl = (*points)[l].p;
                double phi_l = interp_phi(pl.x, pl.y);
#endif
                double d = (pc - pl).norm_L2();

                double sigma_l;

#ifdef P4_TO_P8
                if(phi_l<0) sigma_l = (*mu_m)(pl.x, pl.y, pl.z);
                else        sigma_l = (*mu_p)(pl.x, pl.y, pl.z);
#else
                if(phi_l<0) sigma_l = (*mu_m)(pl.x, pl.y);
                else        sigma_l = (*mu_p)(pl.x, pl.y);
#endif

#ifdef P4_TO_P8
                double Smn = 0.5*((*Smm)(pc.x, pc.y, pc.z) + (*Smm)(pl.x, pl.y, pl.z));
#else
                double Smn = 0.5*((*Smm)(pc.x, pc.y) + (*Smm)(pl.x, pl.y));
#endif

                double sigma_harmonic = sigma_n*sigma_l/(sigma_n + sigma_l+2*sigma_n*sigma_l*dt/(Cm+dt*Smn)/d);

                if(phi_n*phi_l<0)
                {
#ifdef P4_TO_P8
                    Point3 p_ln = (pc+pl)/2;
#else
                    Point2 p_ln = (pc+pl)/2;
#endif

#ifdef P4_TO_P8
                    rhs_p[n] += s*sigma_harmonic* SIGN(phi_n) * (*u_jump)(p_ln.x, p_ln.y, p_ln.z)*Cm/((Cm+dt*Smn)*d/2);
#else
                    rhs_p[n] += s*sigma_harmonic* SIGN(phi_n) * (*u_jump)(p_ln.x, p_ln.y)*Cm/((Cm+dt*Smn)*d/2);
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
                if(fabs(pc.x-xyz_min[0])<EPS && ( (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp += 2*EPS;
                if(fabs(pc.x-xyz_max[0])<EPS && ( (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) x_tmp -= 2*EPS;
                if(fabs(pc.y-xyz_min[1])<EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp += 2*EPS;
                if(fabs(pc.y-xyz_max[1])<EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_00m  || (*points)[l].n==WALL_00p) ) y_tmp -= 2*EPS;
                if(fabs(pc.z-xyz_min[2])<EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp += 2*EPS;
                if(fabs(pc.z-xyz_max[2])<EPS && ( (*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00 || (*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0) ) z_tmp -= 2*EPS;
                rhs_p[n] += s*sigma_n * bc->wallValue(x_tmp, y_tmp, z_tmp);
#else
                if(fabs(pc.x-xyz_min[0])<EPS && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp += 2*EPS;
                if(fabs(pc.x-xyz_max[0])<EPS && ((*points)[l].n==WALL_0m0  || (*points)[l].n==WALL_0p0)) x_tmp -= 2*EPS;
                if(fabs(pc.y-xyz_min[1])<EPS && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp += 2*EPS;
                if(fabs(pc.y-xyz_max[1])<EPS && ((*points)[l].n==WALL_m00  || (*points)[l].n==WALL_p00)) y_tmp -= 2*EPS;
                rhs_p[n] += s*sigma_n * bc->wallValue(x_tmp, y_tmp);
#endif
            }
        }
    }

    ierr = VecRestoreArray(rhs, &rhs_p); CHKERRXX(ierr);

    if (matrix_has_nullspace)
        ierr = MatNullSpaceRemove(A_null_space, rhs, NULL); CHKERRXX(ierr);

    ierr = PetscLogEventEnd(log_PoissonSolverNodeBasedJump_rhsvec_setup, rhs, 0, 0, 0); CHKERRXX(ierr);
}


/*
double my_p4est_electroporation_solve_t::interpolate_solution_from_voronoi_to_tree_on_node_n(p4est_locidx_t n) const
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
    // first check if the node is a voronoi point
    for(unsigned int m=0; m<grid2voro[n].size(); ++m)
    {
        pm = voro_points[grid2voro[n][m]];
        if((pn-pm).norm_L2()<EPS)
        {
            double retval = sol_voro_p[grid2voro[n][m]];
            ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);
            return retval;
        }
    }

    double *phi_p;
    ierr = VecGetArray(phi, &phi_p); CHKERRXX(ierr);

    // if not a grid point, gather all the neighbor voro points and find the
    // three closest with the same sign for phi
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
    // FIXME: What is this?
    // std::cerr << "my_p4est_electroporation_solve_t->interpolate_solution_from_voronoi_to_tree: point is double !" << std::endl;

    // now find the two voronoi points closest to the node
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
                // if point is not already in the list
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
                    }
                }
            }
        }
    }

#ifdef P4_TO_P8
    Point3 p0(voro_points[ni[0]]);
    Point3 p1(voro_points[ni[1]]);
#else
    Point2 p0(voro_points[ni[0]]);
    Point2 p1(voro_points[ni[1]]);
#endif

    // find a third point forming a non-flat triangle
    for(unsigned int k=0; k<ngbd_quads.size(); ++k)
    {
        for(int dir=0; dir<P4EST_CHILDREN; ++dir)
        {
            p4est_locidx_t n_idx = nodes->local_nodes[P4EST_CHILDREN*ngbd_quads[k] + dir];
            for(unsigned int m=0; m<grid2voro[n_idx].size(); ++m)
            {
                // if point is not already in the list
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
                    }
                }
            }
        }
    }

#ifdef P4_TO_P8
    Point3 p2(voro_points[ni[2]]);
#else
    Point2 p2(voro_points[ni[2]]);
#endif

#ifdef P4_TO_P8
    // in 3D, found a fourth point to form a non-flat tetrahedron
    for(unsigned int k=0; k<ngbd_quads.size(); ++k)
    {
        for(int dir=0; dir<P4EST_CHILDREN; ++dir)
        {
            p4est_locidx_t n_idx = nodes->local_nodes[P4EST_CHILDREN*ngbd_quads[k] + dir];
            for(unsigned int m=0; m<grid2voro[n_idx].size(); ++m)
            {
                // if point is not already in the list
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
                    }
                }
            }
        }
    }

    Point3 p3(voro_points[ni[3]]);
#endif

    // make sure we found 3 points
#ifdef P4_TO_P8
    if(di[0]==DBL_MAX || di[1]==DBL_MAX || di[2]==DBL_MAX || di[3]==DBL_MAX)
#else
    if(di[0]==DBL_MAX || di[1]==DBL_MAX || di[2]==DBL_MAX)
#endif
    {
        std::cerr << "my_p4est_electroporation_solve_t->interpolate_solution_from_voronoi_to_tree: not enough points found." << std::endl;
        double retval = sol_voro_p[ni[0]];
        ierr = VecRestoreArray(phi     , &phi_p     ); CHKERRXX(ierr);
        ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);
        return retval;
    }

#ifdef P4_TO_P8
    if(ni[0]==ni[1] || ni[0]==ni[2] || ni[0]==ni[3] || ni[1]==ni[2] || ni[1]==ni[3] || ni[2]==ni[3])
#else
    if(ni[0]==ni[1] || ni[0]==ni[2] || ni[1]==ni[2])
#endif
        std::cerr << "my_p4est_electroporation_solve_t->interpolate_solution_from_voronoi_to_tree: point is double !" << std::endl;

    double f0 = sol_voro_p[ni[0]];
    double f1 = sol_voro_p[ni[1]];
    double f2 = sol_voro_p[ni[2]];
#ifdef P4_TO_P8
    double f3 = sol_voro_p[ni[3]];
#endif

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


//      solving A*C = F,
//          | x0 y0 z0 1 |      | c0 |      | f0 |
//          | x1 y1 z1 1 |      | c1 |      | f1 |
//      A = | x2 y2 z2 1 |, C = | c2 |, F = | f2 |
//          | x3 y3 z3 1 |      | c3 |      | f3 |

//                    | b00 b01 b02 b03 |       | c0 |        | f0 |
//       -1           | b10 b11 b12 b13 |       | c1 |    -1  | f1 |
//      A   = 1/det * | b20 b21 b22 b23 |, and  | c2 | = A  * | f2 |
//                    | b30 b31 b32 b33 |       | c3 |        | f3 |



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

    double c0 = (b00*f0 + b01*f1 + b02*f2 + b03*f3) / det;
    double c1 = (b10*f0 + b11*f1 + b12*f2 + b13*f3) / det;
    double c2 = (b20*f0 + b21*f1 + b22*f2 + b23*f3) / det;
    double c3 = (b30*f0 + b31*f1 + b32*f2 + b33*f3) / det;
    return c0*pn.x + c1*pn.y + c2*pn.z + c3;

//    p0 -= pn; //Raphael's
//    p1 -= pn;
//    p2 -= pn;
//    p3 -= pn;

//    double den  = ((p1-p0).cross(p2-p0)).dot(p3-p0);
//    double c0   = (p1.cross(p2)).dot(p3)/den;
//    double c1   = (p2.cross(p0)).dot(p3)/den;
//    double c2   = (p0.cross(p1)).dot(p3)/den;
//    double c3   = 1.0-c0-c1-c2;

//    return (f0*c0+f1*c1+f2*c2+f3*c3);
#else

    double c0 = ( (p1.y* 1- 1*p2.y)*f0 + ( 1*p2.y-p0.y* 1)*f1 + (p0.y* 1- 1*p1.y)*f2 ) / det;
    double c1 = ( ( 1*p2.x-p1.x* 1)*f0 + (p0.x* 1- 1*p2.x)*f1 + ( 1*p1.x-p0.x* 1)*f2 ) / det;
    double c2 = ( (p1.x*p2.y-p2.x*p1.y)*f0 + (p2.x*p0.y-p0.x*p2.y)*f1 + (p0.x*p1.y-p1.x*p0.y)*f2 ) / det;

    return c0*pn.x + c1*pn.y + c2;

#endif
}
*/

double my_p4est_electroporation_solve_t::interpolate_solution_from_voronoi_to_tree_on_node_n(p4est_locidx_t n) const
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
        if((pn-pm).norm_L2()<sqrt(SQR(xyz_max[0]-xyz_min[0])+SQR(xyz_max[1]-xyz_min[1])
                          #ifdef P4_TO_P8
                                  +SQR(xyz_max[2]-xyz_min[2])
                          #endif
                                  )*EPS)
        {
            double retval = sol_voro_p[grid2voro[n][m]];
            ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);
            return retval;
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
    // FIXME: What is this?
    // std::cerr << "my_p4est_poisson_jump_nodes_voronoi_t->interpolate_solution_from_voronoi_to_tree: point is double !" << std::endl;

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
                    }
                }
            }
        }
    }

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
                    }
                }
            }
        }
    }

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
                    }
                }
            }
        }
    }

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
        double retval = sol_voro_p[ni[0]];
        ierr = VecRestoreArray(phi     , &phi_p     ); CHKERRXX(ierr);
        ierr = VecRestoreArray(sol_voro, &sol_voro_p); CHKERRXX(ierr);
        return retval;
    }

#ifdef P4_TO_P8
    if(ni[0]==ni[1] || ni[0]==ni[2] || ni[0]==ni[3] || ni[1]==ni[2] || ni[1]==ni[3] || ni[2]==ni[3])
#else
    if(ni[0]==ni[1] || ni[0]==ni[2] || ni[1]==ni[2])
#endif
        std::cerr << "my_p4est_poisson_jump_nodes_voronoi_t->interpolate_solution_from_voronoi_to_tree: point is double !" << std::endl;

    double f0 = sol_voro_p[ni[0]];
    double f1 = sol_voro_p[ni[1]];
    double f2 = sol_voro_p[ni[2]];
#ifdef P4_TO_P8
    double f3 = sol_voro_p[ni[3]];
#endif

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

    //  double b00 =  ( p1.y*p2.z + p2.y*p3.z + p3.y*p1.z - p1.y*p3.z - p2.y*p1.z - p3.y*p2.z );
    //  double b01 = -( p0.y*p2.z + p2.y*p3.z + p3.y*p0.z - p0.y*p3.z - p2.y*p0.z - p3.y*p2.z );
    //  double b02 =  ( p0.y*p1.z + p1.y*p3.z + p3.y*p0.z - p0.y*p3.z - p1.y*p0.z - p3.y*p1.z );
    //  double b03 = -( p0.y*p1.z + p1.y*p2.z + p2.y*p0.z - p0.y*p2.z - p1.y*p0.z - p2.y*p1.z );

    //  double b10 = -( p1.x*p2.z + p2.x*p3.z + p3.x*p1.z - p1.x*p3.z - p2.x*p1.z - p3.x*p2.z );
    //  double b11 =  ( p0.x*p2.z + p2.x*p3.z + p3.x*p0.z - p0.x*p3.z - p2.x*p0.z - p3.x*p2.z );
    //  double b12 = -( p0.x*p1.z + p1.x*p3.z + p3.x*p0.z - p0.x*p3.z - p1.x*p0.z - p3.x*p1.z );
    //  double b13 =  ( p0.x*p1.z + p1.x*p2.z + p2.x*p0.z - p0.x*p2.z - p1.x*p0.z - p2.x*p1.z );

    //  double b20 =  ( p1.x*p2.y + p2.x*p3.y + p3.x*p1.y - p1.x*p3.y - p2.x*p1.y - p3.x*p2.y );
    //  double b21 = -( p0.x*p2.y + p2.x*p3.y + p3.x*p0.y - p0.x*p3.y - p2.x*p0.y - p3.x*p2.y );
    //  double b22 =  ( p0.x*p1.y + p1.x*p3.y + p3.x*p0.y - p0.x*p3.y - p1.x*p0.y - p3.x*p1.y );
    //  double b23 = -( p0.x*p1.y + p1.x*p2.y + p2.x*p0.y - p0.x*p2.y - p1.x*p0.y - p2.x*p1.y );

    //  double b30 = -( p1.x*p2.y*p3.z + p2.x*p3.y*p1.z + p3.x*p1.y*p2.z - p1.x*p3.y*p2.z - p2.x*p1.y*p3.z - p3.x*p2.y*p1.z );
    //  double b31 =  ( p0.x*p2.y*p3.z + p2.x*p3.y*p0.z + p3.x*p0.y*p2.z - p0.x*p3.y*p2.z - p2.x*p0.y*p3.z - p3.x*p2.y*p0.z );
    //  double b32 = -( p0.x*p1.y*p3.z + p1.x*p3.y*p0.z + p3.x*p0.y*p1.z - p0.x*p3.y*p1.z - p1.x*p0.y*p3.z - p3.x*p1.y*p0.z );
    //  double b33 =  ( p0.x*p1.y*p2.z + p1.x*p2.y*p0.z + p2.x*p0.y*p1.z - p0.x*p2.y*p1.z - p1.x*p0.y*p2.z - p2.x*p1.y*p0.z );

    //  double c0 = (b00*f0 + b01*f1 + b02*f2 + b03*f3) / det;
    //  double c1 = (b10*f0 + b11*f1 + b12*f2 + b13*f3) / det;
    //  double c2 = (b20*f0 + b21*f1 + b22*f2 + b23*f3) / det;
    //  double c3 = (b30*f0 + b31*f1 + b32*f2 + b33*f3) / det;

    //  return c0*pn.x + c1*pn.y + c2*pn.z + c3;


    p0 -= pn;
    p1 -= pn;
    p2 -= pn;
    p3 -= pn;

    double den  = ((p1-p0).cross(p2-p0)).dot(p3-p0);
    double c0   = (p1.cross(p2)).dot(p3)/den;
    double c1   = (p2.cross(p0)).dot(p3)/den;
    double c2   = (p0.cross(p1)).dot(p3)/den;
    double c3   = 1.0-c0-c1-c2;

    return (f0*c0+f1*c1+f2*c2+f3*c3);
#else

    double c0 = ( (p1.y* 1- 1*p2.y)*f0 + ( 1*p2.y-p0.y* 1)*f1 + (p0.y* 1- 1*p1.y)*f2 ) / det;
    double c1 = ( ( 1*p2.x-p1.x* 1)*f0 + (p0.x* 1- 1*p2.x)*f1 + ( 1*p1.x-p0.x* 1)*f2 ) / det;
    double c2 = ( (p1.x*p2.y-p2.x*p1.y)*f0 + (p2.x*p0.y-p0.x*p2.y)*f1 + (p0.x*p1.y-p1.x*p0.y)*f2 ) / det;

    return c0*pn.x + c1*pn.y + c2;

#endif
}



void my_p4est_electroporation_solve_t::interpolate_solution_from_voronoi_to_tree(Vec solution) const
{
    PetscErrorCode ierr;

    ierr = PetscLogEventBegin(log_PoissonSolverNodeBasedJump_interpolate_to_tree, phi, sol_voro, solution, 0); CHKERRXX(ierr);


    double *solution_p;
    ierr = VecGetArray(solution, &solution_p); CHKERRXX(ierr);

    /* for debugging, compute the error on the voronoi mesh */
    // bousouf
    if(1)
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



void my_p4est_electroporation_solve_t::write_stats(const char *path) const
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




void my_p4est_electroporation_solve_t::print_voronoi_VTK(const char* path) const
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
    double *vn_p, *sol_voro_p;
    // VecGetArray(vn_voro, &vn_p);
    VecGetArray(sol_voro, &sol_voro_p);
    bool periodic[P4EST_DIM] = {false, false, false};
    Voronoi3D::print_VTK_Format(voro, name, xyz_min, xyz_max, periodic, sol_voro_p, sol_voro_p, num_local_voro);
    // VecRestoreArray(vn_voro, &vn_p);
    VecRestoreArray(sol_voro, &sol_voro_p);
    // Voronoi3D::print_VTK_Format(voro, name, xyz_min, xyz_max, periodic);
#else

    Voronoi2D::print_VTK_Format(voro, name);
#endif

}


void my_p4est_electroporation_solve_t::check_voronoi_partition() const
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
                            break;
                        }
                    }

                    if(ok==false)
                    {
#ifdef P4_TO_P8
                        std::cout << p4est->mpirank << " found bad voronoi cell for point # " << n << " : " << (*points)[m].n << ", \t surface = " << (*points)[m].s <<  ", \t Centered on : " << voro[n].get_Center_Point();
#else
                        std::cout << p4est->mpirank << " found bad voronoi cell for point # " << n << " : " << (*points)[m].n << ", \t Centered on : " << voro[n].get_Center_Point();
#endif
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
                throw std::invalid_argument("my_p4est_electroporation_solve_t->check_voronoi_partition: asked to check a non local point or a wall.");

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
