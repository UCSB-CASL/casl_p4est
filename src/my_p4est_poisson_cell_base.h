#ifndef MY_P4EST_POISSON_CELL_BASE_H
#define MY_P4EST_POISSON_CELL_BASE_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolating_function.h>
#include <src/my_p8est_utils.h>
#include <p8est_nodes.h>
#else
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolating_function.h>
#include <src/my_p4est_utils.h>
#include <p4est_nodes.h>
#endif

class PoissonSolverCellBase
{
  const my_p4est_cell_neighbors_t *cell_neighbors_;
  const my_p4est_node_neighbors_t *node_neighbors_;
  typedef my_p4est_cell_neighbors_t::quad_info_t quad_info_t;

  // p4est objects
  p4est_t *p4est_;
  p4est_nodes_t *nodes_;
  p4est_ghost_t *ghost_;

  my_p4est_brick_t *myb_;
  InterpolatingFunctionNodeBase phi_interp;
#ifdef P4_TO_P8
  const CF_3* phi_cf;
#else
  const CF_2* phi_cf;
#endif

  double mu_, diag_add_;
  bool is_matrix_ready;
  bool matrix_has_nullspace;
  double dx_min, dy_min, d_min, diag_min;
#ifdef P4_TO_P8
  double dz_min;
#endif
#ifdef P4_TO_P8
  BoundaryConditions3D *bc_;
#else
  BoundaryConditions2D *bc_;
#endif

  // PETSc objects
  Mat A;
  MatNullSpace A_null_space;
  Vec rhs_, phi_, add_, phi_xx_, phi_yy_;
#ifdef P4_TO_P8
  Vec phi_zz_;
#endif
  bool is_phi_dd_owned;
  KSP ksp;
  PetscErrorCode ierr;

  void preallocate_matrix();
  void setup_negative_laplace_matrix();
  void setup_negative_laplace_rhsvec();

  inline double phi_cell(p4est_locidx_t q, double *phi_ptr) const {
    double p_c = 0;
    for (short i = 0; i<P4EST_CHILDREN; i++)
      p_c += phi_ptr[nodes_->local_nodes[q*P4EST_CHILDREN + i]];
    return (p_c/(double)P4EST_CHILDREN);
  }

  // disallow copy ctr and copy assignment
  PoissonSolverCellBase(const PoissonSolverCellBase& other);
  PoissonSolverCellBase& operator=(const PoissonSolverCellBase& other);

public:
  PoissonSolverCellBase(const my_p4est_cell_neighbors_t *cell_neighbors, const my_p4est_node_neighbors_t* node_neighbors);
  ~PoissonSolverCellBase();

#ifdef P4_TO_P8
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif
  inline void set_rhs(Vec rhs)                 {rhs_      = rhs;}
  inline void set_diagonal(double add)         {diag_add_ = add; is_matrix_ready = false;}
  inline void set_diagonal(Vec add)            {add_      = add; is_matrix_ready = false;}
#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc) {bc_       = &bc; is_matrix_ready = false;}
#else
  inline void set_bc(BoundaryConditions2D& bc) {bc_       = &bc; is_matrix_ready = false;}
#endif
  inline void set_mu(double mu)                {mu_       = mu;  is_matrix_ready = false;}

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
};
#endif // MY_P4EST_POISSON_CELL_BASE_H
