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

class my_p4est_poisson_cells_t
{
  const my_p4est_cell_neighbors_t *ngbd_c;
  const my_p4est_node_neighbors_t *ngbd_n;

  // p4est objects
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;

  my_p4est_brick_t *myb;
  my_p4est_interpolation_nodes_t phi_interp;
#ifdef P4_TO_P8
  const CF_3* phi_cf;
#else
  const CF_2* phi_cf;
#endif

  double mu, diag_add;
  bool is_matrix_ready;
  int matrix_has_nullspace;
  double dx_min, dy_min, d_min, diag_min;
#ifdef P4_TO_P8
  double dz_min;
#endif
#ifdef P4_TO_P8
  BoundaryConditions3D *bc;
#else
  BoundaryConditions2D *bc;
#endif

  // PETSc objects
  Mat A;
  MatNullSpace A_null_space;
  Vec rhs, phi, add, phi_xx, phi_yy;
#ifdef P4_TO_P8
  Vec phi_zz;
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

  // disallow copy ctr and copy assignment
  my_p4est_poisson_cells_t(const my_p4est_poisson_cells_t& other);
  my_p4est_poisson_cells_t& operator=(const my_p4est_poisson_cells_t& other);

public:
  my_p4est_poisson_cells_t(const my_p4est_cell_neighbors_t *ngbd_c, const my_p4est_node_neighbors_t* ngbd_n);
  ~my_p4est_poisson_cells_t();

#ifdef P4_TO_P8
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL);
#else
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL);
#endif
  inline void set_rhs(Vec rhs)                 {this->rhs      = rhs;}
  inline void set_diagonal(double add)         {this->diag_add = add; is_matrix_ready = false;}
  inline void set_diagonal(Vec add)            {this->add      = add; is_matrix_ready = false;}
#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc) {this->bc       = &bc; is_matrix_ready = false;}
#else
  inline void set_bc(BoundaryConditions2D& bc) {this->bc       = &bc; is_matrix_ready = false;}
#endif
  inline void set_mu(double mu)                {this->mu       = mu;  is_matrix_ready = false;}

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
};
#endif // MY_P4EST_POISSON_CELL_BASE_H
