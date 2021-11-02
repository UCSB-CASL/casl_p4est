#ifndef MY_P4EST_POISSON_FACES_H
#define MY_P4EST_POISSON_FACES_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_utils.h>
#include <src/my_p8est_faces.h>
#include <src/voronoi3D.h>
#else
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_utils.h>
#include <src/my_p4est_faces.h>
#include <src/voronoi2D.h>
#endif

class my_p4est_poisson_faces_t
{
  const my_p4est_faces_t *faces;
  const p4est_t *p4est;
  const my_p4est_cell_neighbors_t *ngbd_c;
  const my_p4est_node_neighbors_t *ngbd_n;

  double xyz_min[P4EST_DIM];
  double xyz_max[P4EST_DIM];
  double dxyz[P4EST_DIM];
  double tree_dimensions[P4EST_DIM];
  bool periodic[P4EST_DIM];

  my_p4est_interpolation_nodes_t interp_phi;
  Vec phi;

  Vec *rhs;

  bool apply_hodge_second_derivative_if_neumann; // to possibly fix potential issues with neumann boundary conditions on face dofs aligned with walls
  bool matrix_is_ready[P4EST_DIM], only_diag_is_modified[P4EST_DIM];
  /* control flags within the solve steps:
   * - no matrix update required if matrix_is_ready == true
   * - only an update for diagonal term(s) if matrix_is_ready == false but only_diag_is_modified == true
   * - entire matrix update if both matrix_is_ready and only_diag_is_modified are false
   */
  bool ksp_is_set_from_options[P4EST_DIM], pc_is_set_from_options[P4EST_DIM];

  double current_diag[P4EST_DIM], desired_diag[P4EST_DIM];
  double mu;

  const BoundaryConditionsDIM *bc, *bc_hodge;
  Vec *dxyz_hodge;
  Vec *face_is_well_defined;

  bool compute_partition_on_the_fly;
  vector<Voronoi_DIM> voro[P4EST_DIM];

  Mat A[P4EST_DIM];
  MatNullSpace A_null_space[P4EST_DIM];
  Vec null_space[P4EST_DIM];
  KSP ksp[P4EST_DIM];

  int matrix_has_nullspace[P4EST_DIM];

  void preallocate_matrix(const u_char &dir);

#ifndef P4_TO_P8
  void clip_voro_cell_by_interface(Voronoi2D &voro_cell, const p4est_locidx_t &f_idx, const u_char &dir);
#endif

  void setup_linear_system(const u_char &dir);

  // disallow copy ctr and copy assignment
  my_p4est_poisson_faces_t(const my_p4est_poisson_faces_t& other);
  my_p4est_poisson_faces_t& operator=(const my_p4est_poisson_faces_t& other);

  void setup_linear_solver(int dim, bool use_nonzero_initial_guess, KSPType ksp_type, PCType pc_type);

  inline void reset_current_diag(const u_char &dir)
  {
    current_diag[dir] = 0.0;
  }

  inline bool current_diag_is_as_desired(const u_char &dir) const
  {
    return (fabs(current_diag[dir] - desired_diag[dir]) < EPS*MAX(fabs(current_diag[dir]), fabs(desired_diag[dir])) || (fabs(current_diag[dir]) < EPS && fabs(desired_diag[dir]) < EPS));
  }

public:
  my_p4est_poisson_faces_t(const my_p4est_faces_t *faces, const my_p4est_node_neighbors_t *ngbd);
  ~my_p4est_poisson_faces_t();

  void set_second_order_hodge_correction_on_neumann(bool flag_value){ apply_hodge_second_derivative_if_neumann = flag_value; }

  void set_phi(Vec phi);// if phi is changed, the linear system should be reset...

  void set_rhs(Vec *rhs);

  void set_diagonal(double add);

  void set_mu(double mu_);

  // if the type of bcs is changed, the linear system should be reset...
  void set_bc(const BoundaryConditionsDIM *bc_, Vec *dxyz_hodge_, Vec *face_is_well_defined_, const BoundaryConditionsDIM *bc_hodge_ = NULL);

  void set_compute_partition_on_the_fly(bool do_it_on_the_fly);

  inline const int* get_matrix_has_nullspace() { return matrix_has_nullspace; }

  void solve(Vec *solution, bool use_nonzero_initial_guess=false, KSPType ksp_type=KSPBCGS, PCType pc_type=PCSOR);

  void print_partition_VTK(const char *file, const u_char &dir);

  void global_volume_of_voronoi_tesselation(double voro_global_volume[P4EST_DIM]) const;
};

#endif // MY_P4EST_POISSON_FACES_H
