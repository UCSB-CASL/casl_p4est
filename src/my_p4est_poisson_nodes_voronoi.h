#ifndef MY_P4EST_POISSON_NODES_VORONOI_H
#define MY_P4EST_POISSON_NODES_VORONOI_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_utils.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_utils.h>
#include <src/voronoi2D.h>
#endif

class my_p4est_poisson_nodes_voronoi_t
{
  // p4est objects
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb;
  const my_p4est_node_neighbors_t *ngbd;
  my_p4est_interpolation_nodes_t phi_interp;
  my_p4est_interpolation_nodes_t robin_coef_interp;

  double mu, diag_add;
  bool is_matrix_computed;
  int matrix_has_nullspace;
  double dxyz_min[P4EST_DIM];
  double d_min, diag_min;
  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
#ifdef P4_TO_P8
  BoundaryConditions3D *bc;
#else
  BoundaryConditions2D *bc;
#endif
  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  // PETSc objects
  Mat A;
  p4est_gloidx_t fixed_value_idx_g;
  p4est_gloidx_t fixed_value_idx_l;
  bool is_phi_dd_owned;
  Vec rhs, phi, add, phi_xx[P4EST_DIM];
  Vec robin_coef;
  KSP ksp;
  PetscErrorCode ierr;

  void construct_voronoi_cell(Voronoi2D &voro, p4est_locidx_t n, double *phi_p);

  void preallocate_matrix();

  void setup_negative_laplace_matrix();
  void setup_negative_laplace_rhsvec();

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_voronoi_t(const my_p4est_poisson_nodes_voronoi_t& other);
  my_p4est_poisson_nodes_voronoi_t& operator=(const my_p4est_poisson_nodes_voronoi_t& other);

public:
  my_p4est_poisson_nodes_voronoi_t(const my_p4est_node_neighbors_t *ngbd);
  ~my_p4est_poisson_nodes_voronoi_t();

#ifdef P4_TO_P8
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif
  inline void set_rhs(Vec rhs)                 {this->rhs = rhs;}
  inline void set_diagonal(double add)         {diag_add = add;          is_matrix_computed = false;}
  inline void set_diagonal(Vec add)            {this->add = add;          is_matrix_computed = false;}
#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc) {this->bc = &bc;          is_matrix_computed = false;}
#else
  inline void set_bc(BoundaryConditions2D& bc) {this->bc = &bc;          is_matrix_computed = false;}
#endif
  inline void set_robin_coef(Vec robin_coef)
  {
    this->robin_coef = robin_coef;
    is_matrix_computed = false;
    robin_coef_interp.set_input(robin_coef, linear);
  }
  inline void set_mu(double mu)                {this->mu       = mu;           is_matrix_computed = false;}
  inline void set_is_matrix_computed(bool is_matrix_computed) { this->is_matrix_computed = is_matrix_computed; }
  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace; }

  void shift_to_exact_solution(Vec sol, Vec uex);

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR);
};

#endif // MY_P4EST_POISSON_NODES_VORONOI_H
