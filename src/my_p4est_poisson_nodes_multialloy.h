#ifndef MY_P4EST_POISSON_NODES_MULTIALLOY_H
#define MY_P4EST_POISSON_NODES_MULTIALLOY_H

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

class my_p4est_poisson_nodes_multialloy_t
{
  const my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t           *p4est;
  p4est_nodes_t     *nodes;
  p4est_ghost_t     *ghost;
  my_p4est_brick_t  *myb_;

  double dx_min, dy_min, dz_min, d_min, diag_min;

#ifdef P4_TO_P8
  BoundaryConditions3D *bc_t;
  std::vector<BoundaryConditions3D> *bc_c;
#else
  BoundaryConditions2D *bc_t;
  std::vector<BoundaryConditions2D> *bc_c;
#endif

  std::vector<PetscInt> global_node_offset;
  std::vector<PetscInt> petsc_gloidx;

  // PETSc objects
  Mat A_t;
  std::vector<Mat> A_c;

  Vec RHS_t;
  std::vector<Vec> RHS_c;

  Vec phi, phi_xx, phi_yy, phi_zz;
  bool is_phi_dd_owned;

  KSP ksp_t;
  std::vector<KSP> ksp_c;

  std::vector<bool> node_near_interface;

  PetscErrorCode ierr;

  // parameters
  double dt;
  double lambda;
  double *Dl;
  double *kp;

  CF_2 *vn_cf;
  CF_2 *jump_t;

  Vec rhs_t;
  std::vector<Vec> rhs_c;

  Vec scalling_t;
  std::vector<Vec> scalling_c;

  Vec mask;


  void preallocate_matrices();

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_multialloy_t(const my_p4est_poisson_nodes_t& other);
  my_p4est_poisson_nodes_multialloy_t& operator=(const my_p4est_poisson_nodes_t& other);

public:
  struct interface_point_t
  {
    int dir;
    double dist;
    double w;
    double value;
    double value2;

    interface_point_t(int dir_, double dist_, double w_) {dir = dir_; dist = dist_; w = w_;}
  };

  std::vector< std::vector<interface_point_t> > pointwise_c0_gamma;

  my_p4est_poisson_nodes_multialloy_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_multialloy_t();

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
  // set normal velocity (= jump in Tn and robin coeff for C_i, i > 0) (cf)
  void set_normal_velocity(CF_2& vn_cf_) {vn_cf = &vn_cf_;}

  // set diffusitivities
  void set_dt(double dt_) {dt = dt_;}
  void set_t_mu(double lambda_) {lambda = lambda_;}
  void set_c_mu(double* Dl_) {Dl = Dl_;}

  // set bc on the boundary of the computational box
  void set_t_bc(BoundaryConditions2D& bc_t_) {bc_t = &bc_t_;}
  void set_c_bc(std::vector<BoundaryConditions2D>& bc_c_) {bc_c = &bc_c_;}

  // set rhs (Vec)
  void set_t_rhs(Vec rhs_tm_, Vec rhs_tp_) {rhs_tm = rhs_tm_; rhs_tp = rhs_tp_;}
  void set_c_rhs(std::vector<Vec>& rhs_c_) {rhs_c = &rhs_c_;}

  // specific for temperature (cf)
  void set_t_jump(CF_2& jump_t_) {jump_t = &jump_t_;}

  // specific for c-robin
  void set_c_kp(double* kp_) {kp = kp_;}

  void set_up_matrix_c0();
  void set_up_rhs_c0();

  void set_up_matrix_t();
  void set_up_rhs_t();

  void set_up_matrix_ci();
  void set_up_rhs_ci();


  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  void solve_t(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
  void solve_c(int comp_idx, Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);

  Vec get_mask() { return mask; }
};

#endif // MY_P4EST_POISSON_NODES_MULTIALLOY_H
