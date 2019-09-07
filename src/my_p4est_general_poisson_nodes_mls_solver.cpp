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
void my_p4est_general_poisson_nodes_mls_solver_t::get_residual_and_set_as_rhs(const Vec& psi_hat_)
{

    P4EST_ASSERT(psi_hat_ != NULL && rhs_ != NULL);
    // level-set functions
    bdry_.get_arrays();
    infc_.get_arrays();

    ierr = MatMult(A_, psi_hat_, rhs_); CHKERRXX(ierr);
    double *residual_p = NULL;
    const double *psi_hat_read_only_p = NULL;

    ierr = VecGetArray(rhs_, &residual_p); CHKERRXX(ierr);

    double *add_n = NULL;
    ierr = VecGetArray(rhs_,&add_n);  CHKERRXX(ierr);

    ierr = VecGetArrayRead(psi_hat_, &psi_hat_read_only_p); CHKERRXX(ierr);

    ierr = VecGetArray(rhs_,   &rhs_ptr);   CHKERRXX(ierr);
    ierr = VecGetArray(rhs_m_, &rhs_m_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p_, &rhs_p_ptr); CHKERRXX(ierr);
    foreach_local_node(n, nodes_)
    {

        double bdry_phi_eff_000 = (bdry_.num_phi == 0) ? -1 : bdry_.phi_eff_ptr[n];

        double infc_phi_eff_000 = (infc_.num_phi == 0) ? -1 : infc_.phi_eff_ptr[n];

        p4est_indep_t *ni   = (p4est_indep_t*)sc_array_index(&nodes_->indep_nodes, n);
        if (is_node_Wall(p4est_, ni) && bdry_phi_eff_000 < 0.)
        {

          double xyz_C[P4EST_DIM];
          node_xyz_fr_n(n, p4est_, nodes_, xyz_C);

          if      (wc_type_->value(xyz_C) == DIRICHLET) {
              residual_p[n]= -residual_p[n] + wc_value_->value(xyz_C);

          }
        }
        double mu_n, rhs_n;
        if (infc_phi_eff_000 < 0)
        {
            add_n[n]=0.0;
        }
        else
        {

            add_n[n] = kappa_sqr*sinh(psi_hat_read_only_p[n]);
        }

        switch (node_scheme_[n])
        {
          case NO_DISCRETIZATION: continue; break;
          case WALL_DIRICHLET: continue; break;
          case WALL_NEUMANN: continue; break;
          case FINITE_DIFFERENCE:
           {
            residual_p[n]= rhs_ptr[n]- residual_p[n] -add_n[n];
           }
          break;
          case FINITE_VOLUME:
          {
            my_p4est_finite_volume_t fv;
            if (finite_volumes_initialized_) fv = bdry_fvs_->at(bdry_node_to_fv_[n]);
            else construct_finite_volume(fv, n, p4est_, nodes_, bdry_phi_cf_, bdry_.opn, integration_order_, cube_refinement_, 1, phi_perturbation_);
            residual_p[n]= rhs_ptr[n]*fv.volume - residual_p[n] -add_n[n]*fv.volume;
          }
          break;

          case IMMERSED_INTERFACE:
          {
            my_p4est_finite_volume_t fv_m;
            my_p4est_finite_volume_t fv_p;
            if (finite_volumes_initialized_) fv_m = infc_fvs_->at(infc_node_to_fv_[n]);
            else construct_finite_volume(fv_m, n, p4est_, nodes_, infc_phi_cf_, infc_.opn, integration_order_, cube_refinement_, 1, phi_perturbation_);

            fv_p.full_cell_volume = fv_m.full_cell_volume;
            fv_p.volume           = fv_m.full_cell_volume - fv_m.volume;

            double volume_cut_cell_m = fv_m.volume;
            double volume_cut_cell_p = fv_p.volume;
            residual_p[n] = rhs_m_ptr[n]*volume_cut_cell_m+ rhs_p_ptr[n]*volume_cut_cell_p- residual_p[n]-(0*volume_cut_cell_m + kappa_sqr*sinh(psi_hat_read_only_p[n])*volume_cut_cell_p);

          }
          break;
        }
      }

    ierr = VecRestoreArrayRead(psi_hat_, &psi_hat_read_only_p); psi_hat_read_only_p = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_, &residual_p); residual_p = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_, &add_n); add_n = NULL; CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_m_, &rhs_m_ptr); CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_p_, &rhs_p_ptr); CHKERRXX(ierr);
    ierr = VecGhostUpdateBegin(rhs_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
    ierr = VecGhostUpdateEnd(rhs_, INSERT_VALUES, SCATTER_FORWARD); CHKERRXX(ierr);
}
void my_p4est_general_poisson_nodes_mls_solver_t::get_residual_and_set_as_rhs_v1(const Vec& psi_hat_)
{
    bdry_.get_arrays();
    infc_.get_arrays();
    Vec rhs_m_copy;
    Vec rhs_p_copy;
    ierr = VecDuplicate(rhs_m_, &rhs_m_copy); CHKERRXX(ierr);
    ierr = VecDuplicate(rhs_p_, &rhs_p_copy); CHKERRXX(ierr);
    ierr = VecCopy(rhs_m_, rhs_m_copy); CHKERRXX(ierr);
    ierr = VecCopy(rhs_p_, rhs_p_copy); CHKERRXX(ierr);
    double *rhs_m_copy_ptr;
    double *rhs_p_copy_ptr;
    ierr = VecGetArray(rhs_m_copy,&rhs_m_copy_ptr);CHKERRXX(ierr);
    ierr = VecGetArray(rhs_p_copy,&rhs_p_copy_ptr);CHKERRXX(ierr);
    double *diag_m_temp_ptr;
    double *diag_p_temp_ptr;
    ierr = VecGetArray(diag_m_,&diag_m_temp_ptr); CHKERRXX(ierr);
    ierr = VecGetArray(diag_p_,&diag_p_temp_ptr); CHKERRXX(ierr);
    double const *psi_hat_read_only_ptr;
    ierr = VecGetArrayRead(psi_hat_,&psi_hat_read_only_ptr);
    //ierr = PetscPrintf(p4est_->mpicomm, "line 149 ok "); CHKERRXX(ierr);
    foreach_node(n,nodes_){
        //ierr = PetscPrintf(p4est_->mpicomm, "line 151 ok "); CHKERRXX(ierr);
        double infc_phi_eff_000 = (infc_.num_phi == 0) ? -1 : infc_.phi_eff_ptr[n];
        //ierr = PetscPrintf(p4est_->mpicomm, "line 153 ok "); CHKERRXX(ierr);


        if (infc_phi_eff_000 < 0)
                {
                    //ierr = PetscPrintf(p4est_->mpicomm, "line 154 ok "); CHKERRXX(ierr);
                    //cout << " diag m term " << diag_m_temp_ptr[n] <<'\n';
                    rhs_m_copy_ptr[n]= rhs_m_copy_ptr[n]- diag_m_temp_ptr[n]*sinh(psi_hat_read_only_ptr[n])+diag_m_temp_ptr[n]*cosh(psi_hat_read_only_ptr[n])*psi_hat_read_only_ptr[n];
                }
                else
                {
                    //ierr = PetscPrintf(p4est_->mpicomm, "line 159 ok "); CHKERRXX(ierr);
                    //cout << " diag m term " << diag_m_temp_ptr[n] <<'\n';
                    rhs_p_copy_ptr[n]= rhs_p_copy_ptr[n]- diag_p_temp_ptr[n]*sinh(psi_hat_read_only_ptr[n])+diag_p_temp_ptr[n]*cosh(psi_hat_read_only_ptr[n])*psi_hat_read_only_ptr[n];
                }
    }
    ierr = VecRestoreArrayRead(psi_hat_,&psi_hat_read_only_ptr);CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_m_copy,&rhs_m_copy_ptr);CHKERRXX(ierr);
    ierr = VecRestoreArray(rhs_p_copy,&rhs_p_copy_ptr);CHKERRXX(ierr);
    ierr = VecRestoreArray(diag_m_,&diag_m_temp_ptr);CHKERRXX(ierr);
    ierr = VecRestoreArray(diag_p_,&diag_p_temp_ptr);CHKERRXX(ierr);
    double rhs_m_norm,rhs_m_norm_original, rhs_p_norm,rhs_p_norm_original;

    ierr = VecNorm(rhs_m_copy, NORM_2, &rhs_m_norm); CHKERRXX(ierr);
    ierr = VecNorm(rhs_p_copy, NORM_2, &rhs_p_norm); CHKERRXX(ierr);
    ierr = VecNorm(rhs_m_, NORM_2, &rhs_m_norm_original); CHKERRXX(ierr);
    ierr = VecNorm(rhs_p_, NORM_2, &rhs_p_norm_original); CHKERRXX(ierr);
//    cout << " rhs m " << rhs_m_norm;
//    cout << " rhs p " << rhs_p_norm;
//    cout << " rhs m original" << rhs_m_norm_original;
//    cout << " rhs p original" << rhs_p_norm_original;
    set_rhs(rhs_m_copy,rhs_p_copy);
    //ierr = VecDestroy(rhs_m_copy);CHKERRXX(ierr);
    //ierr = VecDestroy(rhs_p_copy);CHKERRXX(ierr);
}

int     my_p4est_general_poisson_nodes_mls_solver_t::solve_nonlinear(Vec psi_hat,double upper_bound_residual, int it_max, bool validation_flag)
{
    parStopWatch *log_timer = NULL, *solve_subtimer = NULL;
    //ierr = PetscPrintf(p4est_->mpicomm, "line 168 ok "); CHKERRXX(ierr);
    //finalize geometry
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


    psi_hat_is_set=false;
    int iter = 0;


    if (psi_hat_is_set && !validation_flag)
      return iter;
    if(solve_subtimer != NULL)
      solve_subtimer->start("Start the 1st iteration of the solver");
    //ierr = PetscPrintf(p4est_->mpicomm, "line 197 ok "); CHKERRXX(ierr);
    solve(psi_hat);
    //ierr = PetscPrintf(p4est_->mpicomm, "line 199 ok "); CHKERRXX(ierr);
    if(solve_subtimer != NULL){
      solve_subtimer->stop();solve_subtimer->read_duration();
    }
    iter++;
    Vec diag_m_original_copy;
    Vec diag_p_original_copy;
    ierr = VecDuplicate(diag_m_, &diag_m_original_copy); CHKERRXX(ierr);
    ierr = VecDuplicate(diag_p_, &diag_p_original_copy); CHKERRXX(ierr);
    ierr = VecCopy(diag_m_, diag_m_original_copy); CHKERRXX(ierr);
    ierr = VecCopy(diag_p_, diag_p_original_copy); CHKERRXX(ierr);
    Vec diag_m_copy;
    Vec diag_p_copy;
    ierr = VecDuplicate(diag_m_, &diag_m_copy); CHKERRXX(ierr);
    ierr = VecDuplicate(diag_p_, &diag_p_copy); CHKERRXX(ierr);
    ierr = VecCopy(diag_m_, diag_m_copy); CHKERRXX(ierr);
    ierr = VecCopy(diag_p_, diag_p_copy); CHKERRXX(ierr);
    ierr = VecScale(diag_m_copy, -1); CHKERRXX(ierr);
    ierr = VecScale(diag_p_copy, -1); CHKERRXX(ierr);
    set_diag(diag_m_copy, diag_p_copy);

    //ierr = VecDestroy(diag_m_copy);CHKERRXX(ierr);
    //ierr = VecDestroy(diag_p_copy);CHKERRXX(ierr);
    setup_linear_system(false);
    Vec pristine_diag;
    ierr = VecDuplicate(rhs_, &pristine_diag); CHKERRXX(ierr);
    ierr = MatGetDiagonal(A_, pristine_diag); CHKERRXX(ierr);
    //ierr = MatDiagonalSet(A_, pristine_diag, INSERT_VALUES); CHKERRXX(ierr);
    Vec rhs_m_original_copy;
    Vec rhs_p_original_copy;
    ierr = VecDuplicate(rhs_m_, &rhs_m_original_copy); CHKERRXX(ierr);
    ierr = VecDuplicate(rhs_p_, &rhs_p_original_copy); CHKERRXX(ierr);
    ierr = VecCopy(rhs_m_, rhs_m_original_copy); CHKERRXX(ierr);
    ierr = VecCopy(rhs_p_, rhs_p_original_copy); CHKERRXX(ierr);
    get_residual_and_set_as_rhs_v1(psi_hat);
    if(iter < it_max && solve_subtimer != NULL)
    {
       string timer_msg = "Solving nonlinear iterations";
       solve_subtimer->start(timer_msg);
    }

    const double *psi_hat_read_only_ptr = NULL;
    while (iter < it_max)// && two_norm_of_residual > upper_bound_residual)
    {
        Vec diag_m_copy1;
        Vec diag_p_copy1;
        ierr = VecDuplicate(diag_m_, &diag_m_copy1); CHKERRXX(ierr);
        ierr = VecDuplicate(diag_p_, &diag_p_copy1); CHKERRXX(ierr);
        ierr = VecCopy(diag_m_, diag_m_copy1); CHKERRXX(ierr);
        ierr = VecCopy(diag_p_, diag_p_copy1); CHKERRXX(ierr);
        double *diag_m_copy1_ptr;
        double *diag_p_copy1_ptr;
        ierr = VecGetArray(diag_m_copy1,&diag_m_copy1_ptr);CHKERRXX(ierr);
        ierr = VecGetArray(diag_p_copy1,&diag_p_copy1_ptr);CHKERRXX(ierr);
        double *diag_m_temp_ptr;
        double *diag_p_temp_ptr;
        ierr = VecGetArray(diag_m_,&diag_m_temp_ptr);CHKERRXX(ierr);
        ierr = VecGetArray(diag_p_,&diag_p_temp_ptr);CHKERRXX(ierr);
        ierr = VecGetArrayRead(psi_hat,&psi_hat_read_only_ptr);
        foreach_node(n,nodes_)
        {
            double infc_phi_eff_000 = (infc_.num_phi == 0) ? -1 : infc_.phi_eff_ptr[n];
            if (infc_phi_eff_000 < 0)
                    {
                        diag_m_copy1_ptr[n]= diag_m_temp_ptr[n]*cosh(psi_hat_read_only_ptr[n]);
                    }
                    else
                    {
                        diag_p_copy1_ptr[n]= diag_p_temp_ptr[n]*cosh(psi_hat_read_only_ptr[n]);
                    }
        }
        ierr = VecRestoreArray(diag_m_copy1,&diag_m_copy1_ptr);CHKERRXX(ierr);
        ierr = VecRestoreArray(diag_p_copy1,&diag_p_copy1_ptr);CHKERRXX(ierr);
        ierr = VecRestoreArray(diag_m_,&diag_m_temp_ptr);CHKERRXX(ierr);
        ierr = VecRestoreArray(diag_p_,&diag_p_temp_ptr);CHKERRXX(ierr);
        set_diag(diag_m_copy1, diag_p_copy1);
        //ierr = VecDestroy(diag_m_copy);CHKERRXX(ierr);
        //ierr = VecDestroy(diag_p_copy);CHKERRXX(ierr);
        solve(psi_hat);
        ierr = MatDiagonalSet(A_, pristine_diag, INSERT_VALUES); CHKERRXX(ierr);
        set_diag(diag_m_original_copy, diag_p_original_copy);
        set_rhs(rhs_m_original_copy,rhs_p_original_copy);
        get_residual_and_set_as_rhs_v1(psi_hat);
        iter++;
        //ierr = VecDestroy(diag_m_copy1);CHKERRXX(ierr);
        //ierr = VecDestroy(diag_p_copy1);CHKERRXX(ierr);
    }
//    ierr = VecDestroy(diag_m_copy);CHKERRXX(ierr);
//    ierr = VecDestroy(diag_p_copy);CHKERRXX(ierr);
//    ierr = VecDestroy(rhs_m_original_copy);CHKERRXX(ierr);
//    ierr = VecDestroy(rhs_p_original_copy);CHKERRXX(ierr);
//    ierr = VecDestroy(pristine_diag);CHKERRXX(ierr);


//    ierr = MatGetDiagonal(A_, pristine_diag); CHKERRXX(ierr);
//    ierr = MatDiagonalSet(A_, pristine_diag, INSERT_VALUES); CHKERRXX(ierr);

//    double two_norm_of_residual = DBL_MAX;
//    Vec pristine_diag;

//    ierr = VecGetArray(diag_m_, &diag_m_ptr); CHKERRXX(ierr);
//    ierr = VecGetArray(diag_p_, &diag_p_ptr); CHKERRXX(ierr);

//    foreach_node(n, nodes_)
//    {

//        double infc_phi_eff_000 = (infc_.num_phi == 0) ? -1 : infc_.phi_eff_ptr[n];
//        if (infc_phi_eff_000 < 0)
//        {

//            diag_m_ptr[n]= -diag_m_ptr[n];
//        }
//        else
//        {

//            diag_p_ptr[n] =-diag_p_ptr[n];
//        }
//    }
//    ierr = VecRestoreArray(diag_m_, &diag_m_ptr); CHKERRXX(ierr);
//    ierr = VecRestoreArray(diag_p_, &diag_p_ptr); CHKERRXX(ierr);

//    set_diag(diag_m_,diag_p_);



//    ierr= VecDuplicate(rhs_, &pristine_diag);
//    ierr = PetscPrintf(p4est_->mpicomm, "line 230 ok "); CHKERRXX(ierr);
//    setup_linear_system(false);
//    //get_residual_and_set_as_rhs_v1(psi_hat);
//    ierr = PetscPrintf(p4est_->mpicomm, "line 233 ok "); CHKERRXX(ierr);
//    ierr = MatGetDiagonal(A_, pristine_diag); CHKERRXX(ierr);
//    ierr = PetscPrintf(p4est_->mpicomm, "line 235 ok "); CHKERRXX(ierr);
//    double *psi_hat_ptr = NULL;
//    const double *psi_hat_read_only_p = NULL;
//    //Vec delta_psi_hat_p; double *delta_psi_hat_p_ptr; ierr = VecCreateGhostNodes(p4est_, nodes_, &delta_psi_hat_p); CHKERRXX(ierr);
//    ierr = PetscPrintf(p4est_->mpicomm, "line 239 ok "); CHKERRXX(ierr);
//    ierr = VecNorm(rhs_, NORM_2, &two_norm_of_residual); CHKERRXX(ierr);

//    if(iter < it_max && solve_subtimer != NULL)
//    {

//      string timer_msg = "Solving nonlinear iterations";
//      solve_subtimer->start(timer_msg);
//    }

//    while (iter < it_max)// && two_norm_of_residual > upper_bound_residual)
//    {
//        ierr = PetscPrintf(p4est_->mpicomm, "line 251 ok "); CHKERRXX(ierr);
//        ierr = VecGetArray(diag_m_, &diag_m_ptr); CHKERRXX(ierr);
//        ierr = VecGetArray(diag_p_, &diag_p_ptr); CHKERRXX(ierr);
//        ierr = PetscPrintf(p4est_->mpicomm, "line 254 ok "); CHKERRXX(ierr);
//       ierr = VecGetArrayRead(psi_hat,&psi_hat_read_only_p);

//        foreach_node(n, nodes_)
//        {
//            ierr = PetscPrintf(p4est_->mpicomm, "line 259 ok "); CHKERRXX(ierr);
//            double infc_phi_eff_000 = (infc_.num_phi == 0) ? -1 : infc_.phi_eff_ptr[n];
//            if (infc_phi_eff_000 < 0)
//            {
//                ierr = PetscPrintf(p4est_->mpicomm, "line 263 ok "); CHKERRXX(ierr);
//                diag_m_ptr[n]= diag_m_ptr[n]*cosh(psi_hat_read_only_p[n]);;
//            }
//            else
//            {
//                ierr = PetscPrintf(p4est_->mpicomm, "line 268 ok "); CHKERRXX(ierr);
//                diag_p_ptr[n] = diag_p_ptr[n]*cosh(psi_hat_read_only_p[n]);
//            }
//        }
//        ierr = PetscPrintf(p4est_->mpicomm, "line 272 ok "); CHKERRXX(ierr);
//        ierr = VecRestoreArrayRead(psi_hat,&psi_hat_read_only_p);
//        ierr = PetscPrintf(p4est_->mpicomm, "line 274 ok "); CHKERRXX(ierr);
//        ierr = VecRestoreArray(diag_m_, &diag_m_ptr); CHKERRXX(ierr);
//        ierr = PetscPrintf(p4est_->mpicomm, "line 276 ok "); CHKERRXX(ierr);
//        ierr = VecRestoreArray(diag_p_, &diag_p_ptr); CHKERRXX(ierr);
//        ierr = PetscPrintf(p4est_->mpicomm, "line 278 ok "); CHKERRXX(ierr);
//        set_diag(diag_m_,diag_p_);
//        ierr = PetscPrintf(p4est_->mpicomm, "line 280 ok "); CHKERRXX(ierr);
//        setup_linear_system(true);
//        ierr = PetscPrintf(p4est_->mpicomm, "line 285 ok "); CHKERRXX(ierr);
//        solve(psi_hat);
//        ierr = PetscPrintf(p4est_->mpicomm, "line 282 ok "); CHKERRXX(ierr);
//        //set_diag(pristine_diag);
//        ierr = PetscPrintf(p4est_->mpicomm, "line 284 ok "); CHKERRXX(ierr);
//        setup_linear_system(true);
//        ierr = PetscPrintf(p4est_->mpicomm, "line 286 ok "); CHKERRXX(ierr);
        //get_residual_and_set_as_rhs_v1(psi_hat);
        //set_jc(0,zero_cf,zero_cf);
        //ierr = VecNorm(rhs_, NORM_2, &two_norm_of_residual); CHKERRXX(ierr);
//        iter++;
//     }

    if(solve_subtimer != NULL){
      solve_subtimer->stop(); solve_subtimer->read_duration();}

    if(log_timer != NULL)
    {
      log_timer->stop(); log_timer->read_duration();
      delete log_timer; log_timer = NULL;
    }

    psi_hat_is_set=true;
    return iter;
}
