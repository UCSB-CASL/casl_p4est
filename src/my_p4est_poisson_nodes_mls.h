#ifndef MY_P4EST_POISSON_NODES_MLS_H
#define MY_P4EST_POISSON_NODES_MLS_H

#include <petsc.h>

#ifdef P4_TO_P8
#include <p8est.h>
#include <p8est_nodes.h>
#include <src/my_p8est_node_neighbors.h>
#include <src/my_p8est_tools.h>
#include <src/my_p8est_interpolation_nodes.h>
#include <src/my_p8est_utils.h>
#include <src/cube3_mls.h>
#include <src/cube2_mls.h>
#else
#include <p4est.h>
#include <p4est_nodes.h>
#include <src/my_p4est_node_neighbors.h>
#include <src/my_p4est_tools.h>
#include <src/my_p4est_interpolation_nodes.h>
#include <src/my_p4est_utils.h>
#include <src/cube3_mls.h>
#include <src/cube2_mls.h>
#endif



class my_p4est_poisson_nodes_mls_t
{

  const my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t           *p4est;
  p4est_nodes_t     *nodes;
  p4est_ghost_t     *ghost;
  my_p4est_brick_t  *myb_;

  my_p4est_interpolation_nodes_t phi_interp;

  bool    neumann_wall_first_order;
  double  mu_, diag_add_;
  bool    is_matrix_computed;
  int     matrix_has_nullspace;
  double  dx_min, dy_min, d_min, diag_min;
#ifdef P4_TO_P8
  double  dz_min;
#endif

#ifdef P4_TO_P8
  std::vector<BoundaryConditions3D> *bc_;
#else
  std::vector<BoundaryConditions2D> *bc_;
#endif

  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  std::vector<int>        *color_;
  std::vector<action_t>   *action_;

#ifdef P4_TO_P8
  CF_3 *force_;
#else
  CF_2 *force_;
#endif

#ifdef P4_TO_P8
  std::vector<CF_3*>  *phi_cf_;
#else
  std::vector<CF_2*>  *phi_cf_;
#endif


  // PETSc objects
  Mat A;
  p4est_gloidx_t fixed_value_idx_g;
  p4est_gloidx_t fixed_value_idx_l;
  bool is_phi_dd_owned, is_mue_dd_owned;
  Vec rhs_, add_, mue_, mue_xx_, mue_yy_;
  std::vector<Vec> *phi_, *phi_xx_, *phi_yy_;
  std::vector<Vec> *robin_coef_;

  bool keep_scalling;
  Vec scalling;
#ifdef P4_TO_P8
  std::vector<Vec> *phi_zz_;
  Vec mue_zz_;
#endif
  KSP ksp;
  PetscErrorCode ierr;

  void preallocate_matrix();

//  void setup_negative_laplace_matrix_neumann_wall_1st_order();
//  void setup_negative_laplace_rhsvec_neumann_wall_1st_order();

  void setup_negative_laplace_matrix();
  void setup_negative_laplace_rhsvec();

//  void setup_negative_variable_coeff_laplace_matrix();
//  void setup_negative_variable_coeff_laplace_rhsvec();

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_mls_t(const my_p4est_poisson_nodes_mls_t& other);
  my_p4est_poisson_nodes_mls_t& operator=(const my_p4est_poisson_nodes_mls_t& other);

public:

  Vec phi_eff_;

  enum node_loc_t {NODE_INS,NODE_DIR,NODE_MXI,NODE_MXO,NODE_NMN,NODE_OUT};

#ifdef P4_TO_P8
  std::vector<cube3_mls_t> cubes;
#else
  std::vector<cube2_mls_t> cubes;
#endif

  std::vector<node_loc_t>  node_loc;
  my_p4est_poisson_nodes_mls_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_mls_t();

  // inlines setters
  /* FIXME: shouldn't those be references instead of copies ? I guess Vec is just a pointer ... but still ?
   * Mohammad: Vec is just a typedef to _p_Vec* so its merely a pointer under the hood.
   * If you are only passing the vector to access its data its fine to pass it as 'Vec v'
   * However, if 'v' is supposed to change itself, i.e. the the whole Vec object and not just its data
   * then it should either be passed via reference, Vec& v, or pointer, Vec* v, just like
   * any other object
   */
#ifdef P4_TO_P8
  void set_phi(std::vector<Vec> *phi, std::vector<Vec> *phi_xx = NULL, std::vector<Vec> *phi_yy = NULL, std::vector<Vec> *phi_zz = NULL);
#else
  void set_phi(std::vector<Vec> *phi, std::vector<Vec> *phi_xx = NULL, std::vector<Vec> *phi_yy = NULL);
#endif

  inline void set_action(std::vector<action_t>& action) {action_  = &action;}
  inline void set_color(std::vector<int>& color)        {color_   = &color;}

  inline void set_keep_scalling(bool keep_scalling_)    {keep_scalling = keep_scalling_;}

#ifdef P4_TO_P8
  inline void set_force(CF_3 &force) {force_ = &force;}
#else
  inline void set_force(CF_2 &force) {force_ = &force;}
#endif

#ifdef P4_TO_P8
  inline void set_phi_cf(std::vector<CF_3*> &phi_cf) {phi_cf_ = &phi_cf;}
#else
  inline void set_phi_cf(std::vector<CF_2*> &phi_cf) {phi_cf_ = &phi_cf;}
#endif

  inline void set_rhs(Vec rhs)                 {rhs_      = rhs;}
  inline void set_diagonal(double add)         {diag_add_ = add;  is_matrix_computed = false;}
  inline void set_diagonal(Vec add)            {add_      = add;  is_matrix_computed = false;}

#ifdef P4_TO_P8
  inline void set_bc(std::vector<BoundaryConditions3D>& bc) {bc_ = &bc; is_matrix_computed = false;}
#else
  inline void set_bc(std::vector<BoundaryConditions2D>& bc) {bc_ = &bc; is_matrix_computed = false;}
#endif

  inline void set_robin_coef(std::vector<Vec>& robin_coef)      {robin_coef_ = &robin_coef; is_matrix_computed = false;}
  inline void set_mu(double mu)                                 {mu_       = mu;            is_matrix_computed = false;}
  inline void set_is_matrix_computed(bool is_matrix_computed)   { this->is_matrix_computed = is_matrix_computed; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }


  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace; }

  inline void set_first_order_neumann_wall( bool val ) { neumann_wall_first_order=val; }

  void shift_to_exact_solution(Vec sol, Vec uex);

#ifdef P4_TO_P8
  void set_mu(Vec mu, Vec mu_xx = NULL, Vec mu_yy = NULL, Vec mu_zz = NULL);
#else
  void set_mu(Vec mu, Vec mu_xx = NULL, Vec mu_yy = NULL);
#endif

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);

  bool is_inside(int n) {if (node_loc[n] == NODE_INS || node_loc[n] == NODE_NMN) return true; else return false;}
//  bool is_inside(int n) {if (node_loc[n] == NODE_NMN) return true; else return false;}
//  bool is_inside(int n) {if (node_loc[n] == NODE_INS) return true; else return false;}

  std::vector<double> node_vol;
  double node_volume(int n) {return node_vol[n];}
  //
  void construct_domain();
};

#endif // MY_P4EST_POISSON_NODES_MLS_H
