#ifndef MY_P4EST_POISSON_NODES_EXTENDED_H
#define MY_P4EST_POISSON_NODES_EXTENDED_H

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

class my_p4est_poisson_nodes_extended_t
{
  typedef struct
  {
    double val;
    PetscInt n;
  } mat_entry_t;

  const my_p4est_node_neighbors_t *node_neighbors_;

  // p4est objects
  p4est_t           *p4est_;
  p4est_nodes_t     *nodes_;
  p4est_ghost_t     *ghost_;
  my_p4est_brick_t  *myb_;

  // PETSc objects
  Mat A_;
  p4est_gloidx_t fixed_value_idx_g_;
  p4est_gloidx_t fixed_value_idx_l_;
  KSP ksp_;
  PetscErrorCode ierr;
  std::vector<PetscInt> global_node_offset_;
  std::vector<PetscInt> petsc_gloidx_;

  // Geometry
  Vec phi_;
  Vec phi_xx_;
  Vec phi_yy_;
  Vec phi_zz_;

  bool is_phi_dd_owned_;

  double dxyz_m_[P4EST_DIM];
  double dx_min_, dy_min_, d_min_, diag_min_;
#ifdef P4_TO_P8
  double dz_min_;
#endif

  // Equation
  double mu_, diag_add_scalar_;
  Vec diag_add_;
  Vec rhs_;
  Vec mue_;
  Vec mue_xx_, mue_yy_, mue_zz_;

  bool variable_mu_;
  bool is_mue_dd_owned_;

  // Some flags
  bool is_matrix_computed_;
  int matrix_has_nullspace_;
  bool use_pointwise_dirichlet_;
  bool new_pc_;
  bool update_ghost_after_solving_;

#ifdef P4_TO_P8
  BoundaryConditions3D *bc_;
#else
  BoundaryConditions2D *bc_;
#endif

  std::vector<double> scalling_;
  bool keep_scalling_;

  int extrapolation_order_;

  void compute_phi_dd();
  void compute_mue_dd();

  void setup_linear_system(bool setup_matrix, bool setup_rhs);

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_extended_t(const my_p4est_poisson_nodes_extended_t& other);
  my_p4est_poisson_nodes_extended_t& operator=(const my_p4est_poisson_nodes_extended_t& other);

public:
  my_p4est_poisson_nodes_extended_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_extended_t();


#ifdef P4_TO_P8
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL, Vec phi_zz = NULL)
#else
  void set_phi(Vec phi, Vec phi_xx = NULL, Vec phi_yy = NULL)
#endif
  {
    phi_ = phi;

    if (phi_xx != NULL &&
    #ifdef P4_TO_P8
        phi_zz != NULL &&
    #endif
        phi_yy != NULL)
    {
      phi_xx_ = phi_xx;
      phi_yy_ = phi_yy;
#ifdef P4_TO_P8
      phi_zz_ = phi_zz;
#endif
      is_phi_dd_owned_ = false;
    } else {
      compute_phi_dd_();
      is_phi_dd_owned_ = true;
    }

    is_matrix_computed_ = false;
  }

  inline void set_diag_add(double diag_add_scalar) { diag_add_scalar_  = diag_add_scalar; is_matrix_computed_ = false; }
  inline void set_diag_add(Vec diag_add)           { diag_add_         = diag_add;        is_matrix_computed_ = false; }

  inline void set_mu(double mu) { mu_ = mu; variable_mu_ = false; }

#ifdef P4_TO_P8
  void set_mu(Vec mue, Vec mue_xx = NULL, Vec mue_yy = NULL, Vec mue_zz = NULL)
#else
  void set_mu(Vec mue, Vec mue_xx = NULL, Vec mue_yy = NULL)
#endif
  {
    mue_ = mue;

    if (mue_xx != NULL &&
    #ifdef P4_TO_P8
        mue_zz != NULL &&
    #endif
        mue_yy != NULL)
    {
      mue_xx_ = mue_xx;
      mue_yy_ = mue_yy;
#ifdef P4_TO_P8
      mue_zz_ = mue_zz;
#endif
      is_mue_dd_owned_ = false;
    } else {
      compute_mue_dd_();
      is_mue_dd_owned_ = true;
    }

    is_matrix_computed_ = false;

    variable_mu_ = true;
  }

  void assemble_second_derivative_matrices();

  inline void set_extrapolation_order (int order) { extrapolation_order_ = order; }

  inline void set_rhs(Vec rhs) { rhs_ = rhs; }

#ifdef P4_TO_P8
  inline void set_bc(BoundaryConditions3D& bc) {bc_ = &bc; is_matrix_computed = false;}
#else
  inline void set_bc(BoundaryConditions2D& bc) {bc_ = &bc; is_matrix_computed = false;}
#endif

  inline void set_is_matrix_computed(bool is_matrix_computed) { is_matrix_computed_ = is_matrix_computed; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp_, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace_; }

  inline void set_use_pointwise_dirichlet (bool value) { use_pointwise_dirichlet_   = value; }

  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);

  void assemble_matrix(Vec solution);

  inline Mat get_matrix() { return A_; }

  inline std::vector<double>* get_scalling() { return &scalling_; }

  inline void assemble_rhs_only() { setup_linear_system(false, true); }

  //---------------------------------------------------------------------------------
  // some stuff for pointwise dirichlet
  //---------------------------------------------------------------------------------

  struct interface_point_t
  {
    short dir;
    double dist;
    double value;
    interface_point_t(double dir_, double dist_) {dir = dir_; dist = dist_;}
  };

  std::vector< std::vector<interface_point_t> > pointwise_bc_;

  inline void get_xyz_interface_point(p4est_locidx_t n, short i, double *xyz)
  {
    node_xyz_fr_n(n, p4est_, nodes_, xyz);
    short  dir  = pointwise_bc_[n][i].dir;
    double dist = pointwise_bc_[n][i].dist;

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
    pointwise_bc_[n][i].value = val;
  }

  inline double get_interface_point_value(p4est_locidx_t n, short i)
  {
    return pointwise_bc_[n][i].value;
  }

  // linear interpolation of a Vec at an interface point
  inline double interpolate_at_interface_point(p4est_locidx_t n, short i, double *ptr)
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    short  dir  = pointwise_bc_[n][i].dir;
    double dist = pointwise_bc_[n][i].dist;

    p4est_locidx_t neigh;
    double h;
    switch (dir) {
#ifdef P4_TO_P8
      case 0: neigh = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                       : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp); h = dx_min_; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                       : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp); h = dx_min_; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                       : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp); h = dy_min_; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                       : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp); h = dy_min_; break;
      case 4: neigh = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                       : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp); h = dz_min_; break;
      case 5: neigh = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                       : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp); h = dz_min_; break;
#else
      case 0: neigh = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm; h = dx_min_; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm; h = dx_min_; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm; h = dy_min_; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm; h = dy_min_; break;
#endif
    }

    return (ptr[n]*(h-dist) + ptr[neigh]*dist)/h;
  }

  // quadratic interpolation of a Vec at an interface point
  inline double interpolate_at_interface_point(p4est_locidx_t n, short i, double *ptr, double *ptr_dd[P4EST_DIM])
  {
    const quad_neighbor_nodes_of_node_t qnnn = node_neighbors_->get_neighbors(n);

    short  dir  = pointwise_bc_[n][i].dir;
    double dist = pointwise_bc_[n][i].dist;

    p4est_locidx_t neigh;
    double h;
    short dim =0;
    switch (dir) {
#ifdef P4_TO_P8
      case 0: neigh = qnnn.d_m00_m0==0 ? (qnnn.d_m00_0m==0 ? qnnn.node_m00_mm : qnnn.node_m00_mp)
                                       : (qnnn.d_m00_0m==0 ? qnnn.node_m00_pm : qnnn.node_m00_pp); h = dx_min_; dim = 0; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? (qnnn.d_p00_0m==0 ? qnnn.node_p00_mm : qnnn.node_p00_mp)
                                       : (qnnn.d_p00_0m==0 ? qnnn.node_p00_pm : qnnn.node_p00_pp); h = dx_min_; dim = 0; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_mp)
                                       : (qnnn.d_0m0_0m==0 ? qnnn.node_0m0_pm : qnnn.node_0m0_pp); h = dy_min_; dim = 1; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_mp)
                                       : (qnnn.d_0p0_0m==0 ? qnnn.node_0p0_pm : qnnn.node_0p0_pp); h = dy_min_; dim = 1; break;
      case 4: neigh = qnnn.d_00m_m0==0 ? (qnnn.d_00m_0m==0 ? qnnn.node_00m_mm : qnnn.node_00m_mp)
                                       : (qnnn.d_00m_0m==0 ? qnnn.node_00m_pm : qnnn.node_00m_pp); h = dz_min_; dim = 2; break;
      case 5: neigh = qnnn.d_00p_m0==0 ? (qnnn.d_00p_0m==0 ? qnnn.node_00p_mm : qnnn.node_00p_mp)
                                       : (qnnn.d_00p_0m==0 ? qnnn.node_00p_pm : qnnn.node_00p_pp); h = dz_min_; dim = 2; break;
#else
      case 0: neigh = qnnn.d_m00_m0==0 ? qnnn.node_m00_mm : qnnn.node_m00_pm; h = dx_min_; dim = 0; break;
      case 1: neigh = qnnn.d_p00_m0==0 ? qnnn.node_p00_mm : qnnn.node_p00_pm; h = dx_min_; dim = 0; break;
      case 2: neigh = qnnn.d_0m0_m0==0 ? qnnn.node_0m0_mm : qnnn.node_0m0_pm; h = dy_min_; dim = 1; break;
      case 3: neigh = qnnn.d_0p0_m0==0 ? qnnn.node_0p0_mm : qnnn.node_0p0_pm; h = dy_min_; dim = 1; break;
#endif
    }

    double p_dd = .5*(ptr_dd[dim][n] + ptr_dd[dim][neigh]);
    double p0 = ptr[n];
    double p1 = ptr[neigh];

    return .5*(p0+p1) + (p1-p0)*(dist/h-.5) + .5*p_dd*(dist*dist-dist*h);
  }

  // assemble RHS for Poisson equation with jump conditions and continuous and constant mu
//  void assemble_jump_rhs(Vec rhs_out, CF_2& jump_u, CF_2& jump_un, CF_2& rhs_m, CF_2& rhs_p);
  void assemble_jump_rhs(Vec rhs_out, const CF_2& jump_u, CF_2& jump_un, Vec rhs_m_in = NULL, Vec rhs_p_in = NULL);
};

#endif // MY_P4EST_POISSON_NODES_EXTENDED_H
