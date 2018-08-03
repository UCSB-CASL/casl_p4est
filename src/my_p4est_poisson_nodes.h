#ifndef MY_P4EST_POISSON_NODES_H
#define MY_P4EST_POISSON_NODES_H

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
#endif

class my_p4est_poisson_nodes_t
{
  static const int cube_refinement = 1;
  const my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t *p4est;
  p4est_nodes_t *nodes;
  p4est_ghost_t *ghost;
  my_p4est_brick_t *myb_;
  my_p4est_interpolation_nodes_t phi_interp;
//  my_p4est_interpolation_nodes_t robin_coef_interp;

  bool neumann_wall_first_order;
  double mu_, diag_add_;
  bool is_matrix_computed;
  int matrix_has_nullspace;
  double dxyz_m[P4EST_DIM];
  double dx_min, dy_min, d_min, diag_min;
#ifdef P4_TO_P8
  double dz_min;
#endif
#ifdef P4_TO_P8
  BoundaryConditions3D *bc_;
#else
  BoundaryConditions2D *bc_;
#endif
  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  // PETSc objects
  Mat A;
  p4est_gloidx_t fixed_value_idx_g;
  p4est_gloidx_t fixed_value_idx_l;
  bool is_phi_dd_owned, is_mue_dd_owned;
  Vec rhs_, phi_, add_, mue_, phi_xx_, phi_yy_, mue_xx_, mue_yy_;
//  Vec robin_coef_;
#ifdef P4_TO_P8
  Vec phi_zz_, mue_zz_;
#endif
  KSP ksp;
  PetscErrorCode ierr;

  Vec mask;
  std::vector<double> scalling;
  bool keep_scalling;

  bool variable_mu;

  bool use_refined_cube;
  bool use_pointwise_dirichlet;
  bool use_continuous_stencil;

  bool new_pc;

  void preallocate_matrix();

  void setup_negative_variable_coeff_laplace_matrix();
  void setup_negative_variable_coeff_laplace_rhsvec();

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_t(const my_p4est_poisson_nodes_t& other);
  my_p4est_poisson_nodes_t& operator=(const my_p4est_poisson_nodes_t& other);

public:
  my_p4est_poisson_nodes_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_t();

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
  inline void set_rhs(Vec rhs)                 {rhs_      = rhs;}
  inline void set_diagonal(double add)         {diag_add_ = add;          is_matrix_computed = false;}
  inline void set_diagonal(Vec add)            {add_      = add;          is_matrix_computed = false;}
#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc) {bc_       = &bc;          is_matrix_computed = false;}
#else
  inline void set_bc(BoundaryConditions2D& bc) {bc_       = &bc;          is_matrix_computed = false;}
#endif
//  inline void set_robin_coef(Vec robin_coef)   {robin_coef_ = robin_coef; is_matrix_computed = false;
//                                                robin_coef_interp.set_input(robin_coef, linear);}
  inline void set_mu(double mu)                {mu_       = mu;           is_matrix_computed = false; variable_mu = false;}
  inline void set_is_matrix_computed(bool is_matrix_computed) { this->is_matrix_computed = is_matrix_computed; }
  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace; }

  inline void set_first_order_neumann_wall( bool val ) { neumann_wall_first_order=val; }

  inline void set_use_refined_cube( bool val ) { use_refined_cube=val; }

  inline void set_use_continuous_stencil( bool val ) { use_continuous_stencil=val; }

  inline void set_use_pointwise_dirichlet( bool val ) { use_pointwise_dirichlet=val; }

  void shift_to_exact_solution(Vec sol, Vec uex);

#ifdef P4_TO_P8
  void set_mu(Vec mu, Vec mu_xx = NULL, Vec mu_yy = NULL, Vec mu_zz = NULL);
#else
  void set_mu(Vec mu, Vec mu_xx = NULL, Vec mu_yy = NULL);
#endif

//  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCSOR);
  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);

  void assemble_matrix(Vec solution);

  Vec get_mask() { return mask; }

  inline bool get_use_quadratic_continuous_stencil() { return use_continuous_stencil; }

  //---------------------------------------------------------------------------------
  // some stuff for pointwise dirichlet
  //---------------------------------------------------------------------------------

  struct interface_point_t {
    short dir;
    double dist;
    double value;
    interface_point_t(double dir_, double dist_) {dir = dir_; dist = dist_;}
  };

  std::vector< std::vector<interface_point_t> > pointwise_bc;

  inline void get_xyz_interface_point(p4est_locidx_t n, short i, double *xyz)
  {
    node_xyz_fr_n(n, p4est, nodes, xyz);
    short  dir  = pointwise_bc[n][i].dir;
    double dist = pointwise_bc[n][i].dist;

    switch (dir) {
      case 0: xyz[0] -= dist; break;
      case 1: xyz[0] += dist; break;
      case 2: xyz[1] -= dist; break;
      case 3: xyz[1] += dist; break;
#ifdef P4_TO_P8
      case 4: xyz[2] -= dist; break;
      case 5: xyz[2] += dist; break;
#endif
    }
  }

  inline void set_interface_point_value(p4est_locidx_t n, short i, double val)
  {
    pointwise_bc[n][i].value = val;
  }

  inline double get_interface_point_value(p4est_locidx_t n, short i)
  {
    return pointwise_bc[n][i].value;
  }

  inline double interpolate_at_interface_point(p4est_locidx_t n, short i, double *ptr)
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    short  dir  = pointwise_bc[n][i].dir;
    double dist = pointwise_bc[n][i].dist;

    p4est_locidx_t neigh;
    double h;
    switch (dir) {
#ifdef P4_TO_P8
      case 0: neigh = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                       : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp); h = dx_min; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                       : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp); h = dx_min; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                       : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp); h = dy_min; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                       : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp); h = dy_min; break;
      case 4: neigh = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                       : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp); h = dz_min; break;
      case 5: neigh = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                       : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp); h = dz_min; break;
#else
      case 0: neigh = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm; h = dx_min; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm; h = dx_min; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm; h = dy_min; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm; h = dy_min; break;
#endif
      default: throw std::runtime_error("Something is wrong\n");
    }

    return (ptr[n]*(h-dist) + ptr[neigh]*dist)/h;
  }

  inline double interpolate_at_interface_point(p4est_locidx_t n, short i, double *ptr, double *ptr_dd[P4EST_DIM])
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    short  dir  = pointwise_bc[n][i].dir;
    double dist = pointwise_bc[n][i].dist;

    p4est_locidx_t neigh;
    double h;
    short dim =0;
    switch (dir) {
#ifdef P4_TO_P8
      case 0: neigh = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                       : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp); h = dx_min; dim = 0; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                       : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp); h = dx_min; dim = 0; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                       : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp); h = dy_min; dim = 1; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                       : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp); h = dy_min; dim = 1; break;
      case 4: neigh = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                       : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp); h = dz_min; dim = 2; break;
      case 5: neigh = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                       : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp); h = dz_min; dim = 2; break;
#else
      case 0: neigh = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm; h = dx_min; dim = 0; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm; h = dx_min; dim = 0; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm; h = dy_min; dim = 1; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm; h = dy_min; dim = 1; break;
#endif
    }

    double p_dd = .5*(ptr_dd[dim][n] + ptr_dd[dim][neigh]);
    double p0 = ptr[n];
    double p1 = ptr[neigh];

    return .5*(p0+p1) + (p1-p0)*(dist/h-.5) + .5*p_dd*(dist*dist-dist*h);
  }

#ifdef P4_TO_P8
  void assemble_jump_rhs(Vec rhs_out, const CF_3& jump_u, CF_3& jump_un, Vec rhs_m_in = NULL, Vec rhs_p_in = NULL);
#else
  void assemble_jump_rhs(Vec rhs_out, const CF_2& jump_u, CF_2& jump_un, Vec rhs_m_in = NULL, Vec rhs_p_in = NULL);
#endif

};

#endif // MY_P4EST_POISSON_NODES_H
