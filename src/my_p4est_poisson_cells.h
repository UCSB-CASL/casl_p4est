#ifndef MY_P4EST_POISSON_CELL_BASE_H
#define MY_P4EST_POISSON_CELL_BASE_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_cell_neighbors.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_utils.h>
#include <p8est_nodes.h>
#else
#include <src/my_p4est_cell_neighbors.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_utils.h>
#include <p4est_nodes.h>
#endif

//#include <lib/algebra/LinearSolver.h>

class my_p4est_poisson_cells_t
{
  const my_p4est_cell_neighbors_t *ngbd_c;
  const my_p4est_node_neighbors_t *ngbd_n;

  // p4est objects
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  my_p4est_brick_t *myb;

  double mu;
  bool is_matrix_ready, only_diag_is_modified;
  /* control flags within the solve steps:
   * - no matrix update required if is_matrix_ready == true
   * - only an update for diagonal term(s) if is_matrix_ready == false but only_diag_is_modified == true
   * - entire matrix update if both is_matrix_ready and only_diag_is_modified are false
   */
  bool desired_diag_locally_built;
  bool ksp_is_set_from_options, pc_is_set_from_options; // control switches within the solve, too
  int matrix_has_nullspace;

  double dxyz_min[P4EST_DIM];
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
  double d_min, diag_min;
#ifdef P4_TO_P8
  double dz_min;
#endif
#ifdef P4_TO_P8
  BoundaryConditions3D *bc;
#else
  BoundaryConditions2D *bc;
#endif

  // PETSc objects
  bool nullspace_use_fixed_point;
  Mat A;
  MatNullSpace A_null_space;
  Vec null_space;
  p4est_gloidx_t fixed_value_idx_g;
  p4est_gloidx_t fixed_value_idx_l;
  Vec current_diag, desired_diag; // owned by solver!
  Vec rhs, phi;
  KSP ksp;
  PetscErrorCode ierr;

  void preallocate_matrix();
  void setup_negative_laplace_matrix();
  void update_matrix_diag_only();
  void setup_negative_laplace_rhsvec();

  inline double phi_cell(p4est_locidx_t q, double *phi_ptr) const {
    double p_c = 0;
    for (short i = 0; i<P4EST_CHILDREN; i++)
      p_c += phi_ptr[nodes->local_nodes[q*P4EST_CHILDREN + i]];
    return (p_c/(double)P4EST_CHILDREN);
  }

  inline p4est_gloidx_t compute_global_index(p4est_locidx_t quad_idx) const
  {
    if(quad_idx<p4est->local_num_quadrants)
      return p4est->global_first_quadrant[p4est->mpirank] + quad_idx;

    const p4est_quadrant_t *quad = (const p4est_quadrant_t*)sc_array_index(&ghost->ghosts, quad_idx-p4est->local_num_quadrants);
    return p4est->global_first_quadrant[quad_find_ghost_owner(ghost, quad_idx-p4est->local_num_quadrants)] + quad->p.piggy3.local_num;
  }

  inline PetscErrorCode reset_current_diag()
  {
    PetscErrorCode iierr = 0;
    if(current_diag != NULL)
      iierr = VecDestroy(current_diag); CHKERRQ(iierr);
    iierr = VecCreateSeq(PETSC_COMM_SELF, p4est->local_num_quadrants, &current_diag); CHKERRQ(iierr);
    iierr = VecSet(current_diag, 0.0); CHKERRQ(iierr);
    return iierr;
  }

  // disallow copy ctr and copy assignment
  my_p4est_poisson_cells_t(const my_p4est_poisson_cells_t& other);
  my_p4est_poisson_cells_t& operator=(const my_p4est_poisson_cells_t& other);

public:
  my_p4est_poisson_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const my_p4est_node_neighbors_t* ngbd_n);
  ~my_p4est_poisson_cells_t();

  /* Default value is false, the nullspace is then removed from the linear system by calling the Petsc procedures.
   * If you choose true, the point with the smallest global index that is computed for is fixed to zero
   */
  inline void set_nullspace_use_fixed_point(bool val) {this->nullspace_use_fixed_point = val;}

  inline void set_phi(Vec phi, const bool solver_needs_reset=true)// if phi is changed, the linear system should be reset... Except if the user knows more.
  {
    this->phi      = phi;
    if(solver_needs_reset)
    {
      is_matrix_ready = false;
      only_diag_is_modified = false;
    }
  }
  inline void set_rhs(Vec rhs)                 {this->rhs      = rhs; }
  inline void set_diagonal(double add)
  {
    if(!desired_diag_locally_built)
    {
      ierr = VecCreateSeq(PETSC_COMM_SELF, p4est->local_num_quadrants, &desired_diag); CHKERRXX(ierr);
      desired_diag_locally_built = true;
    }
    ierr = VecSet(desired_diag, add); CHKERRXX(ierr);
    only_diag_is_modified = is_matrix_ready;
    is_matrix_ready = false;
  }
  inline void set_diagonal(Vec add)
  {
    if(desired_diag_locally_built)
    {
      ierr = VecDestroy(desired_diag); CHKERRXX(ierr);
      desired_diag_locally_built = false;
    }
    desired_diag = add;
    only_diag_is_modified = is_matrix_ready;
    is_matrix_ready = false;
  }
#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc) {this->bc       = &bc; is_matrix_ready = false; only_diag_is_modified = false; }
#else
  inline void set_bc(BoundaryConditions2D& bc) {this->bc       = &bc; is_matrix_ready = false; only_diag_is_modified = false; }
#endif
  inline void set_mu(double mu)
  {
    P4EST_ASSERT(mu > 0.0);
    if(fabs(this->mu - mu) > EPS*MAX(this->mu, mu)) // actual modification of mu
    {
      is_matrix_ready = false;
      only_diag_is_modified = false;
    }
    this->mu       = mu;
  }

  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace; }

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR);
};
#endif // MY_P4EST_POISSON_CELL_BASE_H
