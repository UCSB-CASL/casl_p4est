//#ifndef MY_P4EST_GENERAL_POISSON_NODES_MLS_SOLVER_H
#define MY_P4EST_GENERAL_POISSON_NODES_MLS_SOLVER_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_interpolation_nodes_local.h>
#include <src/my_p8est_poisson_nodes_mls.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_interpolation_nodes_local.h>
#include <src/my_p4est_poisson_nodes_mls.h>
#include <src/my_p4est_utils.h>
#endif

#include <src/mls_integration/cube3_mls.h>
#include <src/mls_integration/cube2_mls.h>

#define DO_NOT_PREALLOCATE

using std::vector;

class my_p4est_general_poisson_nodes_mls_solver_t: public my_p4est_poisson_nodes_mls_t
{
double        kappa_sqr;
protected:
Vec rhs_original_copy;
Vec diag_m_original_copy;
Vec diag_p_original_copy;
Vec diag_m_copy;
Vec diag_p_copy;
Vec sinh_soln;
Vec soln_ghost;
Vec A_0_soln;
Vec A_0_sinh_soln;
Vec A_k_sinh_soln;
my_p4est_general_poisson_nodes_mls_solver_t(const my_p4est_general_poisson_nodes_mls_solver_t& other);
my_p4est_general_poisson_nodes_mls_solver_t& operator=(const my_p4est_general_poisson_nodes_mls_solver_t& other);
public :
my_p4est_general_poisson_nodes_mls_solver_t(const my_p4est_node_neighbors_t *ngbd);
~my_p4est_general_poisson_nodes_mls_solver_t();
int           solve_nonlinear_v1(Vec psi_hat, double upper_bound_residual, int it_max, bool validation_flag);

private :
PetscErrorCode ierr;
bool          psi_hat_is_set;
public:
  static FILE*        log_file;
  static FILE*        timing_file;
  static FILE*        error_file;
};
