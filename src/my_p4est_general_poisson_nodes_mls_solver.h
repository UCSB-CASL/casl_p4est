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
my_p4est_general_poisson_nodes_mls_solver_t(const my_p4est_general_poisson_nodes_mls_solver_t& other);
my_p4est_general_poisson_nodes_mls_solver_t& operator=(const my_p4est_general_poisson_nodes_mls_solver_t& other);
public :
my_p4est_general_poisson_nodes_mls_solver_t(const my_p4est_node_neighbors_t *ngbd);
~my_p4est_general_poisson_nodes_mls_solver_t();
void          get_linear_diagonal_terms(Vec& pristine_diagonal_terms, double value_to_be_subtracted);
void          clean_matrix_diagonal(const Vec& pristine_diagonal_terms);
void          get_residual_and_set_as_rhs(const Vec& psi_hat_);
void          get_residual_and_set_as_rhs_v1(const Vec& psi_hat_);
void          get_rhs_and_add_plus(Vec& rhs_plus, Vec& add_plus);
void          solve_singular_part();
int           solve_nonlinear(Vec psi_hat, double upper_bound_residual, int it_max, bool validation_flag);
void          make_sure_is_node_sampled(Vec &vector);

private :
PetscErrorCode ierr;
bool          psi_hat_is_set;
public:
  static FILE*        log_file;
  static FILE*        timing_file;
  static FILE*        error_file;
};
