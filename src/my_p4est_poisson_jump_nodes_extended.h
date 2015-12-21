#ifndef MY_P4EST_POISSON_JUMP_NODES_EXTENDED_H
#define MY_P4EST_POISSON_JUMP_NODES_EXTENDED_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p4est_interpolation.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_utils.h>
#endif

class my_p4est_poisson_jump_nodes_extended_t
{
  const my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb_;
  my_p4est_interpolation_nodes_t phi_interp;
#ifdef P4_TO_P8
  const CF_3* phi_cf;
#else
  const CF_2* phi_cf;
#endif

  double mue_p_, mue_m_, diag_add_;
  bool is_matrix_computed;
  int matrix_has_nullspace;
  double dx_min, dy_min, d_min, diag_min;
#ifdef P4_TO_P8
  double dz_min;
#endif
#ifdef P4_TO_P8
  BoundaryConditions3D *bc_;
  const CF_3 *ap_cf_, *bp_cf_;
#else
  BoundaryConditions2D *bc_;
  const CF_2 *ap_cf_, *bp_cf_;
#endif
  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  // PETSc objects
  Mat A;
  p4est_gloidx_t fixed_value_idx_g;
  p4est_gloidx_t fixed_value_idx_l;
  bool is_phi_dd_owned, is_mue_dd_owned;
  Vec rhs_, phi_, add_, mue_, phi_xx_, phi_yy_, mue_xx_, mue_yy_;
#ifdef P4_TO_P8
  Vec phi_zz_, mue_zz_;
#endif
  KSP ksp;
  PetscErrorCode ierr;

  void preallocate_matrix();
  void preallocate_row(p4est_locidx_t n, const quad_neighbor_nodes_of_node_t& qnnn, std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz);

  void setup_negative_laplace_matrix();
  void setup_negative_laplace_rhsvec();

  // disallow copy ctr and copy assignment
  my_p4est_poisson_jump_nodes_extended_t(const my_p4est_poisson_jump_nodes_extended_t& other);
  my_p4est_poisson_jump_nodes_extended_t& operator=(const my_p4est_poisson_jump_nodes_extended_t& other);

public:
  my_p4est_poisson_jump_nodes_extended_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_jump_nodes_extended_t();

  // inlines setters
  /* FIXME: shouldn't those be references instead of copies ? I guess Vec is just a pointer ... but still ?
   * Mohammad: Vec is just a typedef to _p_Vec* so its merely a pointer under the hood.
   * If you are only passing the vector to access its data its fine to pass it as 'Vec v'
   * However, if 'v' is supposed to change itself, i.e. the the whole Vec object and not just its data
   * then it should either be passed via reference, Vec& v, or pointer, Vec* v, just like
   * any other object
   */
#ifdef P4_TO_P8
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif
  inline void set_rhs(Vec rhs)                    {rhs_      = rhs;}
  inline void set_diagonal(double add)            {diag_add_ = add;          is_matrix_computed = false;}
  inline void set_diagonal(Vec add)               {add_      = add;          is_matrix_computed = false;}
#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc)    {bc_       = &bc;          is_matrix_computed = false;}
  inline void set_jump(const CF_3& ap, const CF_3& bp) {ap_cf_ = &ap; bp_cf_ = &bp;}
#else
  inline void set_bc(BoundaryConditions2D& bc)    {bc_       = &bc;          is_matrix_computed = false;}
  inline void set_jump(const CF_2& ap, const CF_2& bp) {ap_cf_ = &ap; bp_cf_ = &bp;}
#endif  
  inline void set_mue(double mue_p, double mue_m) {mue_p_ = mue_p; mue_m_ = mue_m; is_matrix_computed = false;}
  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  void shift_to_exact_solution(Vec sol, Vec uex);

#ifdef P4_TO_P8
  void set_mu(Vec mu, Vec mu_xx = NULL, Vec mu_yy = NULL, Vec mu_zz = NULL);
#else
  void set_mu(Vec mu, Vec mu_xx = NULL, Vec mu_yy = NULL);
#endif

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
};

#endif // MY_P4EST_POISSON_JUMP_NODES_EXTENDED_H
