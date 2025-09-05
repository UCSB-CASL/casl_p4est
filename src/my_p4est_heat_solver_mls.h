#ifndef MY_P4EST_HEAT_SOLVER_MLS_H
#define MY_P4EST_HEAT_SOLVER_MLS_H

#include <src/my_p4est_poisson_nodes_mls.h>
#include <petsc.h>
#include <vector>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <functional>

class my_p4est_heat_solver_mls_t {
public:
    enum TimeSteppingMethod {
        BACKWARD_EULER = 0,
        CRANK_NICOLSON = 1,
        BDF2 = 2,
        THETA_METHOD = 3
    };

    // User hook to (re)sample time-dependent coefficients at a given time.
    // You get both "operator time" (t_op: for Î¼,k and matrix assembly) and
    // "rhs time" (t_rhs: for f sampling). You can also set time-dependent BCs.
    //
    // Write the coefficients into the provided Vecs (k_m_orig, k_p_orig, f_m_orig, f_p_orig)
    // and set Î¼ in the Poisson solver (ps.set_mu(...)). You do NOT need to call
    // ps.set_diag()/ps.set_rhs() here â€” the heat solver will push those after it
    // modifies the vectors for the time step.
    std::function<void(double t_op, double t_rhs,
                       my_p4est_poisson_nodes_mls_t& ps,
                       Vec k_m_orig, Vec k_p_orig,
                       Vec f_m_orig, Vec f_p_orig)>
        refresh_coeffs_at_time;

private:
    bool ic_is_set = false;
    bool debug_mode = true;

    my_p4est_poisson_nodes_mls_t* poisson_solver = nullptr;

    // Domain selector
    // SolveRegion solve_region_ = SolveRegion::NEGATIVE;

    // Time stepping
    double dt = 0.01;
    double current_time = 0.0;
    TimeSteppingMethod time_method = BACKWARD_EULER;
    double theta = 1.0;
    int time_step_number = 0;

    // Solution history
    Vec u_current = nullptr;
    Vec u_prev = nullptr;
    Vec u_prev_prev = nullptr;  // for BDF2

    // Mesh references
    p4est_t* p4est = nullptr;
    p4est_nodes_t* nodes = nullptr;
    double* time_ptr = nullptr;   // external pointer to mirror current_time (for BC functors, etc.)

    // Cached â€œoriginalâ€ coefficient fields for this step
    Vec k_m_original = nullptr, k_p_original = nullptr;   // reaction/diag k
    Vec f_m_original = nullptr, f_p_original = nullptr;   // source f

    // Modified per-step coefficients (mass shifts, Î¸ weights) sent to Poisson
    Vec k_m_modified = nullptr, k_p_modified = nullptr;
    Vec f_m_modified = nullptr, f_p_modified = nullptr;

    // scratch
    Vec tmp_Lu_prev = nullptr; // holds L^n u^n when Î¸<1

    bool vectors_allocated = false;

    static inline void destroy_vec(Vec &v) {
        if (v) { PetscErrorCode ierr = VecDestroy(v); (void) ierr; v = nullptr; }
    }

    // ----------------- small helpers (domain-aware) -----------------
    inline bool in_domain_value(double mask_val) const {
        return (mask_val < 0.0);   // computational region = Ï†<0 (Poisson convention)
    }

    // y[i] += alpha * x[i]  only where in_domain(mask[i]) is true
    void axpy_inside(Vec y, double alpha, Vec x, Vec mask) {
        if (alpha == 0.0) return;
        double *yy; const double *xx, *mm;
        PetscInt n; VecGetLocalSize(mask,&n);
        VecGetArray(y,&yy);
        VecGetArrayRead(x,&xx);
        VecGetArrayRead(mask,&mm);
        for (PetscInt i=0;i<n;i++) if (in_domain_value(mm[i])) yy[i] += alpha * xx[i];
        VecRestoreArray(y,&yy);
        VecRestoreArrayRead(x,&xx);
        VecRestoreArrayRead(mask,&mm);
    }

    // z[i] += alpha  only where in_domain(mask[i]) is true
    void shift_inside(Vec z, double alpha, Vec mask) {
        if (alpha == 0.0) return;
        double *zz; const double *mm;
        PetscInt n; VecGetLocalSize(mask,&n);
        VecGetArray(z,&zz);
        VecGetArrayRead(mask,&mm);
        for (PetscInt i=0;i<n;i++) if (in_domain_value(mm[i])) zz[i] += alpha;
        VecRestoreArray(z,&zz);
        VecRestoreArrayRead(mask,&mm);
    }

    // z[i] += alpha * (a[i] * b[i])  only inside
    void axpy_pointwise_mult_inside(Vec z, double alpha, Vec a, Vec b, Vec mask) {
        if (alpha == 0.0) return;
        double *zz; const double *aa, *bb, *mm;
        PetscInt n; VecGetLocalSize(mask,&n);
        VecGetArray(z,&zz);
        VecGetArrayRead(a,&aa);
        VecGetArrayRead(b,&bb);
        VecGetArrayRead(mask,&mm);
        for (PetscInt i=0;i<n;i++) if (in_domain_value(mm[i])) zz[i] += alpha * (aa[i]*bb[i]);
        VecRestoreArray(z,&zz);
        VecRestoreArrayRead(a,&aa);
        VecRestoreArrayRead(b,&bb);
        VecRestoreArrayRead(mask,&mm);
    }

public:
    // ===== ctor/dtor =====
    my_p4est_heat_solver_mls_t(my_p4est_node_neighbors_t* ngbd_n,
                               p4est_t* p4est_in,
                               p4est_nodes_t* nodes_in,
                               double* current_time_ptr = nullptr)
        : dt(0.01), current_time(0.0), time_method(BACKWARD_EULER),
          theta(1.0), time_step_number(0),
          p4est(p4est_in), nodes(nodes_in), time_ptr(current_time_ptr) {

        poisson_solver = new my_p4est_poisson_nodes_mls_t(ngbd_n);
        allocate_vectors();
    }

    ~my_p4est_heat_solver_mls_t() {
        cleanup_vectors();
        delete poisson_solver;
    }


    // ===== passthroughs to Poisson solver (unchanged API) =====
    void add_boundary(mls_opn_t opn, Vec phi, Vec* phi_dd,
                      BoundaryConditionType bc_type, CF_DIM &bc_value, CF_DIM &bc_coeff) {
        poisson_solver->add_boundary(opn, phi, phi_dd, bc_type, bc_value, bc_coeff);
    }
    void add_boundary(mls_opn_t opn, Vec phi,
                      DIM(Vec phi_x, Vec phi_y, Vec phi_z),
                      BoundaryConditionType bc_type, CF_DIM &bc_value, CF_DIM &bc_coeff) {
    #ifdef P4_TO_P8
        poisson_solver->add_boundary(opn, phi, phi_x, phi_y, phi_z, bc_type, bc_value, bc_coeff);
    #else
        poisson_solver->add_boundary(opn, phi, phi_x, phi_y, bc_type, bc_value, bc_coeff);
    #endif
    }
    void add_interface(mls_opn_t opn, Vec phi, Vec* phi_dd,
                       CF_DIM &jc_value, CF_DIM &jc_flux) {
        poisson_solver->add_interface(opn, phi, phi_dd, jc_value, jc_flux);
    }
    void add_interface(mls_opn_t opn, Vec phi,
                       DIM(Vec phi_x, Vec phi_y, Vec phi_z),
                       CF_DIM &jc_value, CF_DIM &jc_flux) {
    #ifdef P4_TO_P8
        poisson_solver->add_interface(opn, phi, phi_x, phi_y, phi_z, jc_value, jc_flux);
    #else
        poisson_solver->add_interface(opn, phi, phi_x, phi_y, jc_value, jc_flux);
    #endif
    }

    void set_mu(double mu_m, double mu_p) { poisson_solver->set_mu(mu_m, mu_p); }
    void set_mu(double mu) { poisson_solver->set_mu(mu); }
    void set_mu(Vec mue_m, Vec mue_p) { poisson_solver->set_mu(mue_m, mue_p); }
    void set_mu(Vec mue) { poisson_solver->set_mu(mue); }
    void set_mu(Vec mue_m, DIM(Vec mue_m_x, Vec mue_m_y, Vec mue_m_z),
                Vec mue_p, DIM(Vec mue_p_x, Vec mue_p_y, Vec mue_p_z)) {
    #ifdef P4_TO_P8
        poisson_solver->set_mu(mue_m, mue_m_x, mue_m_y, mue_m_z,
                               mue_p, mue_p_x, mue_p_y, mue_p_z);
    #else
        poisson_solver->set_mu(mue_m, mue_m_x, mue_m_y,
                               mue_p, mue_p_x, mue_p_y);
    #endif
    }
    void set_mu(Vec mue_m, Vec* mue_m_dd, Vec mue_p, Vec* mue_p_dd) {
        if (mue_m_dd && mue_p_dd) {
        #ifdef P4_TO_P8
            poisson_solver->set_mu(mue_m, mue_m_dd[0], mue_m_dd[1], mue_m_dd[2],
                                   mue_p, mue_p_dd[0], mue_p_dd[1], mue_p_dd[2]);
        #else
            poisson_solver->set_mu(mue_m, mue_m_dd[0], mue_m_dd[1],
                                   mue_p, mue_p_dd[0], mue_p_dd[1]);
        #endif
        } else {
            poisson_solver->set_mu(mue_m, mue_p);
        }
    }

    // reaction k (cache originals!)
    void set_diag(Vec k_m, Vec k_p) {
        if (!vectors_allocated) allocate_vectors();
        VecCopy(k_m, k_m_original); VecCopy(k_p, k_p_original);
        poisson_solver->set_diag(k_m, k_p);
    }
    void set_diag(Vec k) {
        if (!vectors_allocated) allocate_vectors();
        VecCopy(k, k_m_original); VecCopy(k, k_p_original);
        poisson_solver->set_diag(k);
    }
    void set_diag(double k_m, double k_p) {
        if (!vectors_allocated) allocate_vectors();
        VecSet(k_m_original, k_m); VecSet(k_p_original, k_p);
        poisson_solver->set_diag(k_m, k_p);
    }
    void set_diag(double k) {
        if (!vectors_allocated) allocate_vectors();
        VecSet(k_m_original, k); VecSet(k_p_original, k);
        poisson_solver->set_diag(k);
    }

    // source f (cache originals!)
    void set_rhs(Vec f_m, Vec f_p) {
        if (!vectors_allocated) allocate_vectors();
        VecCopy(f_m, f_m_original); VecCopy(f_p, f_p_original);
        poisson_solver->set_rhs(f_m, f_p);
    }
    void set_rhs(Vec f) {
        if (!vectors_allocated) allocate_vectors();
        VecCopy(f, f_m_original); VecCopy(f, f_p_original);
        poisson_solver->set_rhs(f);
    }

    // misc passthroughs
    void set_wc(const WallBCDIM &bc_type, const CF_DIM &bc_value, bool new_submat_main = true) {
        poisson_solver->set_wc(bc_type, bc_value, new_submat_main);
    }
    void set_wc(BoundaryConditionType bc_type, const CF_DIM &bc_value, bool new_submat_main = true) {
        poisson_solver->set_wc(bc_type, bc_value, new_submat_main);
    }

    void set_lip(double v){ poisson_solver->set_lip(v); }
    void set_integration_order(int v){ poisson_solver->set_integration_order(v); }
    void set_cube_refinement(int v){ poisson_solver->set_cube_refinement(v); }
    void set_jump_scheme(int v){ poisson_solver->set_jump_scheme(v); }
    void set_fv_scheme(int v){ poisson_solver->set_fv_scheme(v); }
    void set_use_sc_scheme(bool v){ poisson_solver->set_use_sc_scheme(v); }
    void set_use_taylor_correction(bool v){ poisson_solver->set_use_taylor_correction(v); }
    void set_kink_treatment(bool v){ poisson_solver->set_kink_treatment(v); }
    void set_first_order_neumann_wall(bool v){ poisson_solver->set_first_order_neumann_wall(v); }
    void set_enfornce_diag_scaling(bool v){ poisson_solver->set_enfornce_diag_scaling(v); }
    void set_use_centroid_always(bool v){ poisson_solver->set_use_centroid_always(v); }
    void set_store_finite_volumes(bool v){ poisson_solver->set_store_finite_volumes(v); }
    void set_phi_perturbation(double v){ poisson_solver->set_phi_perturbation(v); }
    void set_domain_rel_thresh(double v){ poisson_solver->set_domain_rel_thresh(v); }
    void set_interface_rel_thresh(double v){ poisson_solver->set_interface_rel_thresh(v); }
    void set_interpolation_method(interpolation_method v){ poisson_solver->set_interpolation_method(v); }
    void set_dirichlet_scheme(int v){ poisson_solver->set_dirichlet_scheme(v); }
    void set_gf_order(int v){ poisson_solver->set_gf_order(v); }
    void set_gf_stabilized(int v){ poisson_solver->set_gf_stabilized(v); }
    void set_gf_thresh(double v){ poisson_solver->set_gf_thresh(v); }

    void set_tolerances(double rtol, int itmax = PETSC_DEFAULT,
                        double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
        poisson_solver->set_tolerances(rtol, itmax, atol, dtol);
    }

    void set_nonlinear_term(double nl_m_coeff, CF_1 &nl_m, CF_1 &nl_m_prime,
                            double nl_p_coeff, CF_1 &nl_p, CF_1 &nl_p_prime) {
        poisson_solver->set_nonlinear_term(nl_m_coeff, nl_m, nl_m_prime,
                                           nl_p_coeff, nl_p, nl_p_prime);
    }
    void set_nonlinear_term(Vec nl_m_coeff, CF_1 &nl_m, CF_1 &nl_m_prime,
                            Vec nl_p_coeff, CF_1 &nl_p, CF_1 &nl_p_prime) {
        poisson_solver->set_nonlinear_term(nl_m_coeff, nl_m, nl_m_prime,
                                           nl_p_coeff, nl_p, nl_p_prime);
    }
    void set_solve_nonlinear_parameters(int method = 1, double itmax = 10,
                                        double change_tol = 1.e-10, double pde_residual_tol = 0) {
        poisson_solver->set_solve_nonlinear_parameters(method, itmax, change_tol, pde_residual_tol);
    }

    void preassemble_linear_system() { poisson_solver->preassemble_linear_system(); }

    // getters used in main
    Vec get_mask()     { return poisson_solver->get_mask(); }
    Vec get_mask_m()   { return poisson_solver->get_mask_m(); }
    Vec get_mask_p()   { return poisson_solver->get_mask_p(); }
    Vec get_areas()    { return poisson_solver->get_areas(); }
    Vec get_areas_m()  { return poisson_solver->get_areas_m(); }
    Vec get_areas_p()  { return poisson_solver->get_areas_p(); }
    Mat get_matrix()   { return poisson_solver->get_matrix(); }
    double get_time_step() const { return dt; }
    Vec get_solution_at_tn() const { return u_prev; }
    TimeSteppingMethod get_time_method() const { return time_method; }
    PetscInt get_global_idx(p4est_locidx_t n) { return poisson_solver->get_global_idx(n); }
    boundary_conditions_t* get_bc(int phi_idx) { return poisson_solver->get_bc(phi_idx); }

    // ===== heat-specific API =====
    void enable_debug(bool on){ debug_mode = on; }

    void set_time_step(double dt_new) { dt = dt_new; }

    void set_time_stepping_method(TimeSteppingMethod method, double theta_val = 0.5) {
        time_method = method;
        switch (method) {
            case BACKWARD_EULER: theta = 1.0;  break;
            case CRANK_NICOLSON: theta = 0.5;  break;
            case THETA_METHOD:   theta = theta_val; break;
            case BDF2:           theta = 1.0;  break; // irrelevant here
        }

        if (time_method == BACKWARD_EULER && std::fabs(theta - 1.0) >= 1e-12) {
            PetscPrintf(PETSC_COMM_WORLD,
                "ERROR: BACKWARD_EULER expects theta=1.0 (got %g)\n", (double) theta);
            throw std::runtime_error("Invalid theta for BACKWARD_EULER");
        }
        if (time_method == CRANK_NICOLSON && std::fabs(theta - 0.5) >= 1e-12) {
            PetscPrintf(PETSC_COMM_WORLD,
                "ERROR: CRANK_NICOLSON expects theta=0.5 (got %g)\n", (double) theta);
            throw std::runtime_error("Invalid theta for CRANK_NICOLSON");
        }

        if (method == BDF2 && !u_prev_prev) {
            PetscErrorCode ierr;
            ierr = VecCreateGhostNodes(p4est, nodes, &u_prev_prev); CHKERRXX(ierr);
            VecSet(u_prev_prev, 0.0);
        }
    }

    // For static problems you can preload originals once
    void set_linear_coefficients(Vec k_m, Vec k_p){
        if (!vectors_allocated) allocate_vectors();
        VecCopy(k_m, k_m_original); VecCopy(k_p, k_p_original);
    }
    void set_source_terms(Vec f_m, Vec f_p){
        if (!vectors_allocated) allocate_vectors();
        VecCopy(f_m, f_m_original); VecCopy(f_p, f_p_original);
    }

    void set_initial_condition(Vec u0) {
        if (!poisson_solver) {
            throw std::runtime_error("heat: poisson_solver is null in set_initial_condition");
        }
        allocate_vectors(); // your existing allocator if any

        // Copy sampled IC into working vectors
        CHKERRXX(VecCopy(u0, u_current));
        CHKERRXX(VecCopy(u0, u_prev));

        // Get domain mask from the underlying Poisson solver
        Vec mask = poisson_solver->get_mask(); // may be NULL on some setups

        if (mask) {
            PetscInt nloc; CHKERRXX(VecGetLocalSize(mask, &nloc));
            double *u_arr = nullptr; const double *m_arr = nullptr;

            // Zero OUTSIDE (mask >= 0)
            CHKERRXX(VecGetArray(u_current, &u_arr));
            CHKERRXX(VecGetArrayRead(mask, &m_arr));
            for (PetscInt i = 0; i < nloc; ++i) {
                if (m_arr[i] >= 0.0) u_arr[i] = 0.0;
            }
            CHKERRXX(VecRestoreArray(u_current, &u_arr));
            CHKERRXX(VecRestoreArrayRead(mask, &m_arr));

            // keep u_prev consistent
            CHKERRXX(VecCopy(u_current, u_prev));

            // Update ghosts
            CHKERRXX(VecGhostUpdateBegin(u_current, INSERT_VALUES, SCATTER_FORWARD));
            CHKERRXX(VecGhostUpdateEnd  (u_current, INSERT_VALUES, SCATTER_FORWARD));
            CHKERRXX(VecGhostUpdateBegin(u_prev,    INSERT_VALUES, SCATTER_FORWARD));
            CHKERRXX(VecGhostUpdateEnd  (u_prev,    INSERT_VALUES, SCATTER_FORWARD));
        } else {
            PetscPrintf(PETSC_COMM_WORLD,
                "[warn] heat: mask is NULL inside set_initial_condition; IC not masked.\n");
        }

        ic_is_set = true; // or your flag
    }


    // optional convenience overload (mirrors Poisson sampling style)
    void set_initial_condition(CF_DIM &u0_cf) {
        if (!vectors_allocated) allocate_vectors();
        Vec u0=nullptr; VecCreateGhostNodes(p4est, nodes, &u0);
        sample_cf_on_nodes(p4est, nodes, u0_cf, u0);
        set_initial_condition(u0);
        VecDestroy(u0);
    }


    void reset_time(double t0 = 0.0) {
        current_time = t0; time_step_number = 0;
        if (time_ptr) *time_ptr = t0;
    }

    // Single step
    void advance_one_time_step() {
        if (!vectors_allocated) allocate_vectors();

        if (time_method == BACKWARD_EULER && std::fabs(theta - 1.0) >= 1e-12) {
            PetscPrintf(PETSC_COMM_WORLD,
                "ERROR: BACKWARD_EULER expects theta=1.0 (got %g)\n", (double) theta);
            throw std::runtime_error("Invalid theta for BACKWARD_EULER");
        }
        if (time_method == CRANK_NICOLSON && std::fabs(theta - 0.5) >= 1e-12) {
            PetscPrintf(PETSC_COMM_WORLD,
                "ERROR: CRANK_NICOLSON expects theta=0.5 (got %g)\n", (double) theta);
            throw std::runtime_error("Invalid theta for CRANK_NICOLSON");
        }

        // Advance "simulation clock": external BC functors can read this
        const double t_n   = current_time;
        const double t_np1 = current_time + dt;
        const double t_mid = current_time + 0.5*dt;

        // Debug (before)
        if (debug_mode && (time_step_number < 3 || time_step_number % 100 == 0)) {
            double norm_before;
            VecNorm(u_current, NORM_2, &norm_before);
            PetscPrintf(PETSC_COMM_WORLD,
                "Step %d: t=%f -> %f, |u_before|=%e\n",
                time_step_number, t_n, t_np1, norm_before);
        }

        // Assemble for the chosen method
        switch (time_method) {
            case BACKWARD_EULER:   setup_theta_family(/*theta=*/1.0,  t_op(t_np1), t_rhs(t_np1)); break;
            case CRANK_NICOLSON:   setup_theta_family(/*theta=*/0.5,  t_op(t_mid), t_rhs(t_mid),  /*need_Lu_n=*/true, t_n); break;
            case THETA_METHOD:     setup_theta_family(/*theta=*/theta,
                                                      t_op( (theta==0.5)?t_mid:t_np1 ),  // common practice
                                                      t_rhs((theta==0.5)?t_mid:t_np1),
                                                      /*need_Lu_n=*/(theta<1.0), t_n);
                                   break;
            case BDF2:             setup_bdf2(t_np1, t_np1); break;
            default: throw std::runtime_error("Unknown time stepping method");
        }

        // For BCs that read current_time via time_ptr, present operator time (typical)
        if (time_ptr) *time_ptr = (time_method==CRANK_NICOLSON || time_method==THETA_METHOD) ?
                                   ((theta==0.5)? t_mid : t_np1) : t_np1;

        // Solve
        poisson_solver->solve(u_current, true);

        // Debug report only
        if (debug_mode && (time_step_number < 3 || time_step_number % 100 == 0)) {
            Vec mask = poisson_solver->get_mask();
            if (mask) {
                const double *u_arr = nullptr, *mask_arr = nullptr;
                VecGetArrayRead(u_current, &u_arr);
                VecGetArrayRead(mask, &mask_arr);

                PetscInt nloc = 0;
                VecGetLocalSize(mask, &nloc);

                int count_inside = 0, count_outside = 0;
                double max_inside = 0, max_outside = 0;

                for (PetscInt i = 0; i < nloc; ++i) {
                    if (in_domain_value(mask_arr[i])) {  // inside
                        ++count_inside;
                        max_inside = std::max(max_inside, std::fabs(u_arr[i]));
                    } else {                              // outside
                        ++count_outside;
                        max_outside = std::max(max_outside, std::fabs(u_arr[i]));
                    }
                }

                VecRestoreArrayRead(u_current, &u_arr);
                VecRestoreArrayRead(mask, &mask_arr);

                PetscPrintf(PETSC_COMM_WORLD,
                    "  After solve: %d inside (max=%e), %d outside (max=%e)\n",
                    count_inside, max_inside, count_outside, max_outside);
            }
        }

        // Update history
        if (u_prev_prev) VecCopy(u_prev, u_prev_prev);
        VecCopy(u_current, u_prev);
        ++time_step_number;
        current_time = t_np1;
    }

    // Loop until T (inclusive up to floating-point tolerance)
    void advance_to_time(double t_final, int /*save_every_steps*/ = -1) {
        while (current_time + 0.5 * dt < t_final) {
            advance_one_time_step();
        }
    }

    // Steady solves passthroughs
    void solve_steady_state(Vec solution, bool use_guess = false) {
        poisson_solver->set_diag(k_m_original, k_p_original);
        poisson_solver->set_rhs(f_m_original, f_p_original);
        poisson_solver->solve(solution, use_guess);
    }
    void solve_steady_state_nonlinear(Vec solution, bool use_guess = false) {
        poisson_solver->set_diag(k_m_original, k_p_original);
        poisson_solver->set_rhs(f_m_original, f_p_original);
        poisson_solver->solve_nonlinear(solution, use_guess);
    }

    // Accessors
    Vec get_solution() { return u_current; }
    double get_current_time() const { return current_time; }
    int get_time_step_number() const { return time_step_number; }

    // ---------- Pointwise BC helpers (pass-throughs) ----------
    int pw_bc_num_value_pts(int phi_idx) { return poisson_solver->pw_bc_num_value_pts(phi_idx); }
    int pw_bc_num_robin_pts(int phi_idx) { return poisson_solver->pw_bc_num_robin_pts(phi_idx); }
    void pw_bc_xyz_value_pt(int phi_idx, int pt_idx, double pt_xyz[]) { poisson_solver->pw_bc_xyz_value_pt(phi_idx, pt_idx, pt_xyz); }
    void pw_bc_xyz_robin_pt(int phi_idx, int pt_idx, double pt_xyz[]) { poisson_solver->pw_bc_xyz_robin_pt(phi_idx, pt_idx, pt_xyz); }

    // ---------- Pointwise JC helpers (pass-throughs) ----------
    int pw_jc_num_taylor_pts(int phi_idx) { return poisson_solver->pw_jc_num_taylor_pts(phi_idx); }
    int pw_jc_num_integr_pts(int phi_idx) { return poisson_solver->pw_jc_num_integr_pts(phi_idx); }
    void pw_jc_xyz_taylor_pt(int phi_idx, int pt_idx, double pt_xyz[]) { poisson_solver->pw_jc_xyz_taylor_pt(phi_idx, pt_idx, pt_xyz); }
    void pw_jc_xyz_integr_pt(int phi_idx, int pt_idx, double pt_xyz[]) { poisson_solver->pw_jc_xyz_integr_pt(phi_idx, pt_idx, pt_xyz); }

    // ---------- Pointwise setters (pass-throughs) ----------
    void set_bc(int phi_idx, BoundaryConditionType bc_type,
                CF_DIM &bc_value, CF_DIM &bc_coeff) {
        poisson_solver->set_bc(phi_idx, bc_type, bc_value, bc_coeff);
    }
    void set_bc(int phi_idx, BoundaryConditionType bc_type,
                std::vector<double> &bc_value_pw,
                std::vector<double> &bc_value_pw_robin,
                std::vector<double> &bc_coeff_pw_robin) {
        poisson_solver->set_bc(phi_idx, bc_type, bc_value_pw, bc_value_pw_robin, bc_coeff_pw_robin);
    }
    void set_jc(int phi_idx, CF_DIM &jc_value, CF_DIM &jc_flux) {
        poisson_solver->set_jc(phi_idx, jc_value, jc_flux);
    }
    void set_jc(int phi_idx,
                std::vector<double> &jc_sol_jump_taylor,
                std::vector<double> &jc_flx_jump_taylor,
                std::vector<double> &jc_flx_jump_integr) {
        poisson_solver->set_jc(phi_idx, jc_sol_jump_taylor, jc_flx_jump_taylor, jc_flx_jump_integr);
    }

    // ---------- Effective level-set getters (pass-throughs) ----------
    Vec get_boundary_phi_eff() { return poisson_solver->get_boundary_phi_eff(); }
    Vec get_interface_phi_eff() { return poisson_solver->get_interface_phi_eff(); }

private:
    // helper structs to pass clear intent
    static double t_op(double t)  { return t; }
    static double t_rhs(double t) { return t; }

    // ===== internals =====
    void allocate_vectors() {
        if (vectors_allocated) return;
        PetscErrorCode ierr;
        ierr = VecCreateGhostNodes(p4est, nodes, &u_current); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &u_prev);    CHKERRXX(ierr);

        ierr = VecCreateGhostNodes(p4est, nodes, &k_m_original); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &k_p_original); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &f_m_original); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &f_p_original); CHKERRXX(ierr);

        ierr = VecCreateGhostNodes(p4est, nodes, &k_m_modified); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &k_p_modified); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &f_m_modified); CHKERRXX(ierr);
        ierr = VecCreateGhostNodes(p4est, nodes, &f_p_modified); CHKERRXX(ierr);

        ierr = VecCreateGhostNodes(p4est, nodes, &tmp_Lu_prev); CHKERRXX(ierr);

        VecSet(u_current, 0.0);
        VecSet(u_prev, 0.0);
        VecSet(k_m_original, 0.0); VecSet(k_p_original, 0.0);
        VecSet(f_m_original, 0.0); VecSet(f_p_original, 0.0);
        VecSet(k_m_modified, 0.0); VecSet(k_p_modified, 0.0);
        VecSet(f_m_modified, 0.0); VecSet(f_p_modified, 0.0);
        VecSet(tmp_Lu_prev, 0.0);

        vectors_allocated = true;
    }

    void cleanup_vectors() {
        if (!vectors_allocated) return;
        destroy_vec(u_current);
        destroy_vec(u_prev);
        destroy_vec(u_prev_prev);
        destroy_vec(k_m_original);
        destroy_vec(k_p_original);
        destroy_vec(f_m_original);
        destroy_vec(f_p_original);
        destroy_vec(k_m_modified);
        destroy_vec(k_p_modified);
        destroy_vec(f_m_modified);
        destroy_vec(f_p_modified);
        destroy_vec(tmp_Lu_prev);
        vectors_allocated = false;
    }

    // --- generic Î¸-family setup ---
    // (u^{n+1}-u^n)/dt = Î¸ L^{op} u^{n+1} + (1-Î¸) L^{n} u^{n} + f^{rhs}
    // â‡’ (L^{op} + 1/(Î¸ dt) I) u^{n+1} = 1/(Î¸ dt) u^n + f^{rhs}/Î¸ + ((1-Î¸)/Î¸) L^{n} u^{n}
    //
    // op_time: time to sample Î¼,k for the operator (and BCs)
    // rhs_time: time to sample f
    // If need_Lu_n, we compute L^n u^n with matrix at time t_n.
    void setup_theta_family(double thet,
                            double op_time, double rhs_time,
                            bool need_Lu_n = false, double t_n_for_L = 0.0)
    {
        Vec mask = poisson_solver->get_mask();
        const double coef_mass = 1.0 / (thet * dt);

        // DEBUG: Check mask statistics
        if (debug_mode && time_step_number < 3) {
            PetscInt n_inside = 0, n_total = 0;
            const double *mask_arr;
            VecGetArrayRead(mask, &mask_arr);
            VecGetLocalSize(mask, &n_total);
            for (PetscInt i = 0; i < n_total; i++) {
                if (in_domain_value(mask_arr[i])) n_inside++;
            }
            VecRestoreArrayRead(mask, &mask_arr);
            PetscPrintf(PETSC_COMM_WORLD,
                "  setup_theta: %d/%d nodes inside domain, coef_mass=%e\n",
                n_inside, n_total, coef_mass);
        }

        // (0) allow user to refresh coefficients at the requested times
        if (refresh_coeffs_at_time) {
            refresh_coeffs_at_time(op_time, rhs_time,
                                   *poisson_solver,
                                   k_m_original, k_p_original,
                                   f_m_original, f_p_original);
        }

        // (1) build L^n u^n in tmp_Lu_prev if needed (CN / Î¸<1 paths)
        if (need_Lu_n && thet < 1.0) {
            // assemble operator at t_n_for_L
            if (refresh_coeffs_at_time) {
                refresh_coeffs_at_time(t_n_for_L, rhs_time,
                                       *poisson_solver,
                                       k_m_original, k_p_original,
                                       f_m_original, f_p_original);
            }
            // tell Poisson about k^n so its matrix reflects L^n
            poisson_solver->set_diag(k_m_original, k_p_original);

            // compute L^n u^n
            Mat A_n = poisson_solver->get_matrix();
            MatMult(A_n, u_prev, tmp_Lu_prev);

            // NOTE: no zero_outside() call here anymore; weâ€™ll only add tmp_Lu_prev
            // to the RHS **inside** via axpy_inside(...) below.
        } else {
            VecSet(tmp_Lu_prev, 0.0);
        }

        // (2) re-assemble operator at op_time for LHS and sample f at rhs_time
        if (refresh_coeffs_at_time) {
            refresh_coeffs_at_time(op_time, rhs_time,
                                   *poisson_solver,
                                   k_m_original, k_p_original,
                                   f_m_original, f_p_original);
        }

        // LHS diagonal: k^{op} + 1/(Î¸ dt) (only inside when a mask exists)
        VecCopy(k_m_original, k_m_modified);
        VecCopy(k_p_original, k_p_modified);
        if (mask) {
            shift_inside(k_m_modified, coef_mass, mask);
            shift_inside(k_p_modified, coef_mass, mask);
        } else {
            VecShift(k_m_modified, coef_mass);
            VecShift(k_p_modified, coef_mass);
        }

        // RHS: f^{rhs}/Î¸ + u^n/(Î¸ dt) + ((1-Î¸)/Î¸) L^n u^n   (again, only inside)
        VecCopy(f_m_original, f_m_modified);
        VecCopy(f_p_original, f_p_modified);

        const double c1 = coef_mass;           // 1/(Î¸ dt)
        const double c2 = 1.0 / thet;          // 1/Î¸
        const double c3 = (1.0 - thet) / thet; // (1-Î¸)/Î¸

        if (mask) {
            // scale f by 1/Î¸ inside
            axpy_inside(f_m_modified, (c2 - 1.0), f_m_original, mask);
            axpy_inside(f_p_modified, (c2 - 1.0), f_p_original, mask);
            // + u^n/(Î¸ dt)
            axpy_inside(f_m_modified, c1, u_prev, mask);
            axpy_inside(f_p_modified, c1, u_prev, mask);
            // + ((1-Î¸)/Î¸) * (L^n u^n)
            if (need_Lu_n && thet < 1.0) {
                axpy_inside(f_m_modified, c3, tmp_Lu_prev, mask);
                axpy_inside(f_p_modified, c3, tmp_Lu_prev, mask);
            }
        } else {
            // no mask => apply everywhere (rare, but keep behavior consistent)
            VecScale(f_m_modified, c2);
            VecScale(f_p_modified, c2);
            VecAXPY(f_m_modified, c1, u_prev);
            VecAXPY(f_p_modified, c1, u_prev);
            if (need_Lu_n && thet < 1.0) {
                VecAXPY(f_m_modified, c3, tmp_Lu_prev);
                VecAXPY(f_p_modified, c3, tmp_Lu_prev);
            }
        }

        // Push modified coefficients to Poisson for the implicit solve
        poisson_solver->set_diag(k_m_modified, k_p_modified);
        poisson_solver->set_rhs(f_m_modified, f_p_modified);

        // DEBUG: Check if k was actually modified
        if (debug_mode && time_step_number < 3) {
            double k_orig_max, k_mod_max;
            VecMax(k_m_original, NULL, &k_orig_max);
            VecMax(k_m_modified, NULL, &k_mod_max);
            PetscPrintf(PETSC_COMM_WORLD,
                "  k modification: %e -> %e (expected add %e)\n",
                k_orig_max, k_mod_max, coef_mass);
        }
    }


    // BDF2:
    // (3u^{n+1}-4u^n+u^{n-1})/(2dt) = L^{op} u^{n+1} + f^{rhs}
    // â‡’ (L^{op} + 3/(2dt) I)u^{n+1} = 2/dt*u^n - 1/(2dt)u^{n-1} + f^{rhs}
    void setup_bdf2(double op_time, double rhs_time) {
        Vec mask = poisson_solver->get_mask();
        const double inv_dt = 1.0/dt;
        const double alpha = 1.5*inv_dt; // 3/(2dt)

        if (refresh_coeffs_at_time) {
            refresh_coeffs_at_time(op_time, rhs_time,
                                   *poisson_solver,
                                   k_m_original, k_p_original,
                                   f_m_original, f_p_original);
        }

        VecCopy(k_m_original, k_m_modified);
        VecCopy(k_p_original, k_p_modified);
        if (mask) {
            shift_inside(k_m_modified, alpha, mask);
            shift_inside(k_p_modified, alpha, mask);
        } else {
            VecShift(k_m_modified, alpha);
            VecShift(k_p_modified, alpha);
        }

        VecCopy(f_m_original, f_m_modified);
        VecCopy(f_p_original, f_p_modified);

        if (mask) {
            axpy_inside(f_m_modified,  2.0*inv_dt, u_prev, mask);
            axpy_inside(f_p_modified,  2.0*inv_dt, u_prev, mask);
            if (u_prev_prev) {
                axpy_inside(f_m_modified, -0.5*inv_dt, u_prev_prev, mask);
                axpy_inside(f_p_modified, -0.5*inv_dt, u_prev_prev, mask);
            }
        } else {
            VecAXPY(f_m_modified,  2.0*inv_dt, u_prev);
            VecAXPY(f_p_modified,  2.0*inv_dt, u_prev);
            if (u_prev_prev) {
                VecAXPY(f_m_modified, -0.5*inv_dt, u_prev_prev);
                VecAXPY(f_p_modified, -0.5*inv_dt, u_prev_prev);
            }
        }

        poisson_solver->set_diag(k_m_modified, k_p_modified);
        poisson_solver->set_rhs(f_m_modified, f_p_modified);
    }

    bool check_steady_state(double tol = 1e-10) {
        double diff;
        VecAXPY(u_current, -1.0, u_prev);  // u_current - u_prev
        VecNorm(u_current, NORM_2, &diff);
        return diff < tol;
    }
};

#endif // MY_P4EST_HEAT_SOLVER_MLS_H