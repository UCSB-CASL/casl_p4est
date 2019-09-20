#ifdef P4_TO_P8
#include "my_p8est_general_poisson_nodes_mls_solver.h"
#include <src/my_p8est_refine_coarsen.h>
#include <src/my_p8est_macros.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#else
#include "my_p4est_general_poisson_nodes_mls_solver.h"
#include <src/my_p4est_refine_coarsen.h>
#include <src/my_p4est_macros.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#endif

#include <src/petsc_compatibility.h>
#include <src/casl_math.h>
#include <src/matrix.h>
#include <src/my_p4est_solve_lsqr.h>
// logging variables -- defined in src/petsc_logging.cpp
#ifndef CASL_LOG_EVENTS
#undef PetscLogEventBegin
#undef PetscLogEventEnd
#define PetscLogEventBegin(e, o1, o2, o3, o4) 0
#define PetscLogEventEnd(e, o1, o2, o3, o4) 0
#else
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_matrix_preallocation;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_setup_linear_system;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize_matrix_and_rhs;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize_matrix;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize_rhs;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_preassemble_linear_system;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_KSPSolve;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_solve;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_compute_finite_volumes;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_compute_finite_volumes_connections;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_determine_node_types;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_discretize;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_submatrix;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_correct_rhs_jump;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_correct_submat_main_jump;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_matrix;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_submat_main;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_assemble_submat_jump;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_add_submat_robin;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_add_submat_diag;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_scale_matrix_by_diagonal;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_scale_rhs_by_diagonal;
extern PetscLogEvent log_my_p4est_poisson_nodes_mls_compute_diagonal_scaling;
#endif
#ifndef CASL_LOG_FLOPS
#undef PetscLogFlops
#define PetscLogFlops(n) 0
#endif

using namespace std;
my_p4est_general_poisson_nodes_mls_solver_t::my_p4est_general_poisson_nodes_mls_solver_t(const my_p4est_node_neighbors_t *ngbd):my_p4est_poisson_nodes_mls_t(ngbd)
{
  kappa_sqr=1.0;
  psi_hat_is_set=false;
}
my_p4est_general_poisson_nodes_mls_solver_t::~my_p4est_general_poisson_nodes_mls_solver_t()
{}
FILE* my_p4est_general_poisson_nodes_mls_solver_t::log_file     = NULL;
FILE* my_p4est_general_poisson_nodes_mls_solver_t::timing_file  = NULL;
FILE* my_p4est_general_poisson_nodes_mls_solver_t::error_file   = NULL;
int     my_p4est_general_poisson_nodes_mls_solver_t::solve_nonlinear(Vec soln,double upper_bound_residual, int it_max, bool validation_flag)
{
    parStopWatch *log_timer = NULL, *solve_subtimer = NULL;

    //finalize geometry
    int iter=0;
    set_boundary_phi_eff (bdry_.phi_eff);
    set_interface_phi_eff(infc_.phi_eff);
    bdry_.get_arrays();
    infc_.get_arrays();

    if( log_file != NULL)
    {
      ierr = PetscFPrintf( p4est_->mpicomm,  log_file, " \n"); CHKERRXX(ierr);
      ierr = PetscFPrintf( p4est_->mpicomm,  log_file, "------------------------------------------------------------------------------- \n"); CHKERRXX(ierr);
      ierr = PetscFPrintf( p4est_->mpicomm,  log_file, "Solving the Poisson-Boltzmann equation on a %d/%d grid with %d proc(s) \n",    p4est_->mpisize); CHKERRXX(ierr);
      if(validation_flag)
      {
        ierr = PetscFPrintf( p4est_->mpicomm,  log_file, "Solving for validation!!! \n", p4est_->mpisize); CHKERRXX(ierr);
      }

      if( timing_file != NULL)
      {
        log_timer = new parStopWatch(parStopWatch::all_timings,  log_file,  p4est_->mpicomm);
        log_timer->start("Resolution of the nonlinear Poisson-Boltzmann Equation");
      }
    }
    if(solve_subtimer != NULL)
      solve_subtimer->start("Start the 1st iteration of the solver");

    solve(soln, 0);
    double norm_of_linear_rhs;
    ierr = VecNorm(rhs_, NORM_2, &norm_of_linear_rhs); CHKERRXX(ierr);


    if(solve_subtimer != NULL){
      solve_subtimer->stop();solve_subtimer->read_duration();
    }
    iter++;
    Vec del_soln; ierr = VecCreateGhostNodes(p4est_, nodes_, &del_soln); CHKERRXX(ierr);
    ierr = VecDuplicate(soln, &del_soln); CHKERRXX(ierr);
    double residual_of_del_soln_norm=DBL_MAX;

        ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_setup_linear_system, 0, 0, 0, 0); CHKERRXX(ierr);
        Mat A_k;
        ierr = MatDuplicate(A_,MAT_COPY_VALUES,&A_k);CHKERRXX(ierr);

        ierr = VecDuplicate(rhs_, &rhs_original_copy); CHKERRXX(ierr);
        ierr = VecCopy(rhs_, rhs_original_copy); CHKERRXX(ierr);
        ierr = VecDuplicate(diag_m_, &diag_m_original_copy); CHKERRXX(ierr);
        ierr = VecDuplicate(diag_p_, &diag_p_original_copy); CHKERRXX(ierr);
        ierr = VecCopy(diag_m_, diag_m_original_copy); CHKERRXX(ierr);
        ierr = VecCopy(diag_p_, diag_p_original_copy); CHKERRXX(ierr);
        ierr = VecDuplicate(diag_m_, &diag_m_copy); CHKERRXX(ierr);
        ierr = VecDuplicate(diag_p_, &diag_p_copy); CHKERRXX(ierr);

        Mat A_0;
        ierr = MatDuplicate(submat_main_,MAT_COPY_VALUES,&A_0);CHKERRXX(ierr);

        if (enfornce_diag_scaling_)
        {
          ierr = PetscLogEventBegin(log_my_p4est_poisson_nodes_mls_scale_matrix_by_diagonal, 0, 0, 0, 0); CHKERRXX(ierr);
          ierr = MatDiagonalScale(A_0, diag_scaling_, NULL); CHKERRXX(ierr);
          ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_scale_matrix_by_diagonal, 0, 0, 0, 0); CHKERRXX(ierr);
        }


        while(iter<it_max && residual_of_del_soln_norm>upper_bound_residual )
        {

        ierr= VecDuplicate(soln,&soln_ghost);CHKERRXX(ierr);

        ierr= MatMultAdd(submat_jump_ghost_,soln,soln,soln_ghost);


        ierr= VecAXPY(soln_ghost,1,rhs_jump_);

        double *diag_m_original_copy_ptr;
        double *diag_p_original_copy_ptr;
        ierr = VecGetArray(diag_m_original_copy,&diag_m_original_copy_ptr);CHKERRXX(ierr);

        ierr = VecGetArray(diag_p_original_copy,&diag_p_original_copy_ptr);CHKERRXX(ierr);

        double *diag_m_copy_ptr;
        double *diag_p_copy_ptr;
        ierr = VecGetArray(diag_m_copy,&diag_m_copy_ptr);CHKERRXX(ierr);
        ierr = VecGetArray(diag_p_copy,&diag_p_copy_ptr);CHKERRXX(ierr);


        double *soln_ptr;
        ierr = VecGetArray(soln,&soln_ptr);CHKERRXX(ierr);
        double *soln_ghost_ptr;

        ierr = VecGetArray(soln_ghost,&soln_ghost_ptr);CHKERRXX(ierr);

        ierr= VecDuplicate(soln,&sinh_soln);CHKERRXX(ierr);
        double *sinh_soln_ptr;
        ierr = VecGetArray(sinh_soln,&sinh_soln_ptr);CHKERRXX(ierr);

        bdry_.get_arrays();
        infc_.get_arrays();
        foreach_node(n,nodes_)
        {

            double infc_phi_eff_000 = (infc_.num_phi == 0) ? -1 : infc_.phi_eff_ptr[n];

            if (infc_phi_eff_000 < 0)
                    {

                        diag_m_copy_ptr[n]= diag_m_original_copy_ptr[n]*cosh(soln_ptr[n]);
                        diag_p_copy_ptr[n]= diag_p_original_copy_ptr[n]*cosh(soln_ghost_ptr[n]);
                        sinh_soln_ptr[n]= sinh(soln_ptr[n]);
                    }
                    else
                    {
                        diag_m_copy_ptr[n]= diag_m_original_copy_ptr[n]*cosh(soln_ghost_ptr[n]);
                        diag_p_copy_ptr[n]= diag_p_original_copy_ptr[n]*cosh(soln_ptr[n]);
                        sinh_soln_ptr[n]= sinh(soln_ptr[n]);
                        //ierr = PetscPrintf(p4est_->mpicomm, "line 1582 ok "); CHKERRXX(ierr);
                    }
        }
        ierr = VecRestoreArray(diag_m_original_copy,&diag_m_original_copy_ptr);CHKERRXX(ierr);
        ierr = VecRestoreArray(diag_p_original_copy,&diag_p_original_copy_ptr);CHKERRXX(ierr);
        ierr = VecRestoreArray(diag_m_copy,&diag_m_copy_ptr);CHKERRXX(ierr);
        ierr = VecRestoreArray(diag_p_copy,&diag_p_copy_ptr);CHKERRXX(ierr);
        ierr = VecRestoreArray(soln_ghost,&soln_ghost_ptr);CHKERRXX(ierr);
        ierr = VecRestoreArray(sinh_soln,&sinh_soln_ptr);CHKERRXX(ierr);
        ierr = VecGetArray(soln,&soln_ptr);CHKERRXX(ierr);
        set_diag(diag_m_copy,diag_p_copy);
        setup_linear_system(false);
        //ierr = PetscPrintf(p4est_->mpicomm, "line 1593 ok "); CHKERRXX(ierr);
        ierr= VecDuplicate(soln,&A_0_soln);CHKERRXX(ierr);
        ierr= VecDuplicate(soln,&A_k_sinh_soln);CHKERRXX(ierr);
        ierr= VecDuplicate(soln,&A_0_sinh_soln);CHKERRXX(ierr);
        //ierr = PetscPrintf(p4est_->mpicomm, "line 1596 ok "); CHKERRXX(ierr);
        ierr= MatMult(A_0, soln, A_0_soln);CHKERRXX(ierr);
        double norm_of_A_0_soln;
        ierr = VecNorm(A_0_soln, NORM_2, &norm_of_A_0_soln); CHKERRXX(ierr);
        //cout<< " norm of A_0_soln-> "<<norm_of_A_0_soln<< "\n";
        //ierr = PetscPrintf(p4est_->mpicomm, "line 1597 ok "); CHKERRXX(ierr);
        ierr= MatMult(A_k, sinh_soln, A_k_sinh_soln); CHKERRXX(ierr);// A(k)*sinh(soln)
        double norm_of_A_k_sinh_soln;
        ierr = VecNorm(A_k_sinh_soln, NORM_2, &norm_of_A_k_sinh_soln); CHKERRXX(ierr);
        //cout<< " norm of A_k_sinh_soln-> "<<norm_of_A_k_sinh_soln<< "\n";
        //ierr = PetscPrintf(p4est_->mpicomm, "line 1598 ok "); CHKERRXX(ierr);
        ierr= MatMult(A_0, sinh_soln, A_0_sinh_soln);CHKERRXX(ierr);
        double norm_of_A_0_sinh_soln;
        ierr = VecNorm(A_0_sinh_soln, NORM_2, &norm_of_A_0_sinh_soln); CHKERRXX(ierr);
        //cout<< " norm of A_0_sinh_soln-> "<<norm_of_A_0_sinh_soln<< "\n";
        //ierr = PetscPrintf(p4est_->mpicomm, "line 1599 ok "); CHKERRXX(ierr);
        ierr= VecAXPY(rhs_,-1,A_0_soln);CHKERRXX(ierr);
        //ierr = PetscPrintf(p4est_->mpicomm, "line 1600 ok "); CHKERRXX(ierr);
        ierr= VecAXPY(rhs_,-1,A_k_sinh_soln);CHKERRXX(ierr);
        //ierr = PetscPrintf(p4est_->mpicomm, "line 1601 ok "); CHKERRXX(ierr);
        ierr= VecAXPY(rhs_,1,A_0_sinh_soln);CHKERRXX(ierr);
        //ierr = PetscPrintf(p4est_->mpicomm, "line 1602 ok "); CHKERRXX(ierr);
        double norm_of_rhs;
        ierr = VecNorm(rhs_, NORM_2, &norm_of_rhs); CHKERRXX(ierr);
        //cout<< " norm of rhs-> "<<norm_of_rhs<< "\n";
        //ierr = PetscPrintf(p4est_->mpicomm, "line 1603 ok "); CHKERRXX(ierr);
        ierr = PetscLogEventEnd(log_my_p4est_poisson_nodes_mls_setup_linear_system, 0, 0, 0, 0); CHKERRXX(ierr);
        bool use_nonzero_guess = false;
        bool update_ghost = true;
        KSPType ksp_type = KSPBCGS;
        PCType pc_type = PCHYPRE;
        invert_linear_system(del_soln, use_nonzero_guess, update_ghost, ksp_type, pc_type);
        iter++;
        ierr = VecNorm(del_soln, NORM_2, &residual_of_del_soln_norm); CHKERRXX(ierr);
        cout<<"residual of del_soln-> "<< residual_of_del_soln_norm<< "\n";
        ierr= VecAXPY(soln,1,del_soln);CHKERRXX(ierr);
        ierr = VecCopy(rhs_original_copy,rhs_); CHKERRXX(ierr);
    }

        ierr= MatDestroy(A_k);CHKERRXX(ierr);

        ierr= MatDestroy(A_0);CHKERRXX(ierr);
    return iter;

}
