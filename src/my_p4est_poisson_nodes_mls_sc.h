#ifndef MY_P4EST_POISSON_NODES_MLS_SC_H
#define MY_P4EST_POISSON_NODES_MLS_SC_H

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

#include <src/mls_integration/cube3_mls.h>
#include <src/mls_integration/cube2_mls.h>

#define DO_NOT_PREALLOCATE

class my_p4est_poisson_nodes_mls_sc_t
{
  struct mat_entry_t
  {
    double val;
    PetscInt n;
    mat_entry_t(PetscInt n=0, double val=0) : n(n), val(val) {}
  };

  // p4est objects
  const my_p4est_node_neighbors_t *node_neighbors_;

  p4est_t           *p4est_;
  p4est_nodes_t     *nodes_;
  p4est_ghost_t     *ghost_;
  my_p4est_brick_t  *myb_;

  // grid variables
  double lip_;
  double diag_min_;
  double d_min_;
  double DIM( dx_min_, dy_min_, dz_min_ );
  double dxyz_m_[P4EST_DIM];

  // PETSc objects
  Mat A_;
  KSP ksp_;
  PetscErrorCode ierr;
  p4est_gloidx_t fixed_value_idx_g_;
  p4est_gloidx_t fixed_value_idx_l_;
  std::vector<PetscInt> global_node_offset_;
  std::vector<PetscInt> petsc_gloidx_;

  // Geometry
  struct geometry_t
  {
    unsigned int num_phi;

    Vec phi_eff;

    std::vector<Vec>    phi;
    std::vector<CF_DIM *> *phi_cf;

    std::vector<Vec> DIM( phi_x,
                          phi_y,
                          phi_z );

    std::vector<Vec> DIM( phi_xx,
                          phi_yy,
                          phi_zz );

    std::vector<mls_opn_t> opn;
    std::vector<int>       clr;

    bool is_phi_eff_owned;
    std::vector<bool> is_phi_d_owned;
    std::vector<bool> is_phi_dd_owned;

    geometry_t() : num_phi(0), phi_eff(NULL), is_phi_eff_owned(0) {}
    ~geometry_t()
    {
      PetscErrorCode ierr;
      if (is_phi_eff_owned) { ierr = VecDestroy(this->phi_eff); CHKERRXX(ierr); }

      for (unsigned int i = 0; i < num_phi; ++i)
      {
        if (is_phi_d_owned[i])
        {
          XCOMP( ierr = VecDestroy(phi_x[i]); CHKERRXX(ierr); );
          YCOMP( ierr = VecDestroy(phi_y[i]); CHKERRXX(ierr); );
          ZCOMP( ierr = VecDestroy(phi_z[i]); CHKERRXX(ierr); );
        }
        if (is_phi_dd_owned[i])
        {
          XCOMP( ierr = VecDestroy(phi_xx[i]); CHKERRXX(ierr); );
          YCOMP( ierr = VecDestroy(phi_yy[i]); CHKERRXX(ierr); );
          ZCOMP( ierr = VecDestroy(phi_zz[i]); CHKERRXX(ierr); );
        }
      }
    }

  } bdry, infc;

  // Boundary, wall and jump conditions
  const WallBC2D *wc_type;
  const CF_DIM   *wc_value;

  std::vector< BoundaryConditionType > bc_type;
  std::vector< CF_DIM *> bc_value;
  std::vector< CF_DIM *> bc_coeff;

  std::vector< CF_DIM *> jc_value;
  std::vector< CF_DIM *> jc_flux;

  // Equation
  Vec   rhs_;
  Vec   rhs_m_;
  Vec   rhs_p_;
  CF_DIM *rhs_cf_;

  Vec    diag_add_m_;
  Vec    diag_add_p_;
  double diag_add_m_scalar_;
  double diag_add_p_scalar_;

  double mu_m_;
  double mu_p_;

  bool variable_mu_;

  bool is_mue_m_dd_owned_;
  bool is_mue_p_dd_owned_;

  Vec  mue_m_, DIM(mue_m_xx_, mue_m_yy_, mue_m_zz_);
  Vec  mue_p_, DIM(mue_p_xx_, mue_p_yy_, mue_p_zz_);

  // solver options
  int integration_order_;
  int cube_refinement_;
  bool use_sc_scheme_;
  bool use_pointwise_dirichlet_;
  bool use_taylor_correction_;
  bool kink_special_treatment_;
  bool update_ghost_after_solving_;
  bool try_remove_hanging_cells_;
  bool neumann_wall_first_order_;
  bool enfornce_diag_scalling_;
  double phi_perturbation_;
  double domain_rel_thresh_;
  double interface_rel_thresh_;
  interpolation_method interp_method_;

  // Some flags
  bool new_pc_;
  bool is_matrix_computed_;
  int matrix_has_nullspace_;

  // auxiliary variables
  Vec mask_;
  Vec areas_;
  Vec areas_m_;
  Vec areas_p_;
//  Vec volumes_;
  Vec node_type_;

  double face_area_scalling_;

  std::vector<double> scalling_;

  bool keep_scalling_;
  bool volumes_computed_;
  bool volumes_owned_;

  Vec exact_;

  enum discretization_scheme_t { FDM, FVM, JUMP } discretization_scheme_;

  int jump_scheme_;

  void compute_volumes();
  void compute_phi_eff(Vec &phi_eff, std::vector<Vec> *phi, std::vector<mls_opn_t> *action, bool &is_phi_eff_owned);
  void compute_phi_dd(std::vector<Vec> *&phi, DIM( std::vector<Vec> *&phi_xx, std::vector<Vec> *&phi_yy, std::vector<Vec> *&phi_zz ), bool &is_phi_dd_owned);
  void compute_phi_d (std::vector<Vec> *&phi, DIM( std::vector<Vec> *&phi_x,  std::vector<Vec> *&phi_y,  std::vector<Vec> *&phi_z  ), bool &is_phi_d_owned);
  void compute_mue_dd();

  double compute_weights_through_face(double A P8C(double B), bool *neighbor_exists_face, double *weights_face, double theta, bool *map_face);

  void preallocate_matrix();

  void setup_linear_system(bool setup_matrix, bool setup_rhs);

  // disallow copy ctr and copy assignment
  my_p4est_poisson_nodes_mls_sc_t(const my_p4est_poisson_nodes_mls_sc_t& other);
  my_p4est_poisson_nodes_mls_sc_t& operator=(const my_p4est_poisson_nodes_mls_sc_t& other);
public:
  my_p4est_poisson_nodes_mls_sc_t(const my_p4est_node_neighbors_t *node_neighbors);
  ~my_p4est_poisson_nodes_mls_sc_t();

  inline PetscInt get_global_idx(p4est_locidx_t n) { return petsc_gloidx_[n]; }
  inline void set_try_remove_hanging_cells(bool value) { try_remove_hanging_cells_ = value; }
  inline void set_phi_cf(std::vector<CF_DIM *> &phi_cf) { bdry.phi_cf = &phi_cf; }
  inline void set_exact(Vec exact) { exact_ = exact; }
  inline void set_lip(double lip) { lip_ = lip; }

  void add_phi(geometry_t &g, mls_opn_t opn, Vec phi, Vec phi_d[], Vec phi_dd[])
  {
    g.opn.push_back(opn);
    g.phi.push_back(phi);
    g.clr.push_back(g.num_phi);

    if (phi_d != NULL)
    {
      g.is_phi_d_owned.push_back(0);

      XCOMP( g.phi_x.push_back(phi_d[0]) );
      YCOMP( g.phi_y.push_back(phi_d[1]) );
      ZCOMP( g.phi_z.push_back(phi_d[2]) );
    }
    else
    {
      g.is_phi_d_owned.push_back(1);

      XCOMP( g.phi_x.push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &g.phi_x.back() ); CHKERRXX(ierr); );
      YCOMP( g.phi_y.push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &g.phi_y.back() ); CHKERRXX(ierr); );
      ZCOMP( g.phi_z.push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &g.phi_z.back() ); CHKERRXX(ierr); );

      node_neighbors_->first_derivatives_central(g.phi.back(), DIM( g.phi_x.back(),
                                                                    g.phi_y.back(),
                                                                    g.phi_z.back() ) );
    }

    if (phi_dd != NULL)
    {
      g.is_phi_dd_owned.push_back(0);

      XCOMP( g.phi_xx.push_back(phi_dd[0]) );
      YCOMP( g.phi_yy.push_back(phi_dd[1]) );
      ZCOMP( g.phi_zz.push_back(phi_dd[2]) );
    }
    else
    {
      g.is_phi_dd_owned.push_back(1);

      XCOMP( g.phi_xx.push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &g.phi_xx.back() ); CHKERRXX(ierr); );
      YCOMP( g.phi_yy.push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &g.phi_yy.back() ); CHKERRXX(ierr); );
      ZCOMP( g.phi_zz.push_back(Vec()); ierr = VecCreateGhostNodes(p4est_, nodes_, &g.phi_zz.back() ); CHKERRXX(ierr); );

      node_neighbors_->second_derivatives_central(g.phi.back(), DIM( g.phi_xx.back(),
                                                                     g.phi_yy.back(),
                                                                     g.phi_zz.back() ) );
    }

    g.num_phi++;
  }

  inline void add_boundary (mls_opn_t opn, Vec phi, Vec phi_d[], Vec phi_dd[], BoundaryConditionType bc_type, CF_DIM &bc_value, CF_DIM &bc_coeff)
  {
    this->bc_type.push_back(bc_type);
    this->bc_value.push_back(&bc_value);
    this->bc_coeff.push_back(&bc_coeff);

    add_phi(bdry, opn, phi, phi_d, phi_dd);
  }

  inline void add_interface(mls_opn_t opn, Vec phi, Vec phi_d[], Vec phi_dd[], CF_DIM &jc_value, CF_DIM &jc_flux)
  {
    this->jc_value.push_back(&jc_value);
    this->jc_flux.push_back(&jc_flux);

    add_phi(infc, opn, phi, phi_d, phi_dd);
  }

  inline void set_boundary_phi_eff(Vec phi_eff)
  {
    if (phi_eff == NULL)
    {
      if (bdry.num_phi == 1) bdry.phi_eff = bdry.phi[0];
      else compute_phi_eff(bdry.phi_eff, &bdry.phi, &bdry.opn, bdry.is_phi_eff_owned);
    } else
      bdry.phi_eff = phi_eff;
  }

  inline void set_interface_phi_eff(Vec phi_eff)
  {
    if (phi_eff == NULL)
    {
      if (infc.num_phi == 1) infc.phi_eff = infc.phi[0];
      else compute_phi_eff(infc.phi_eff, &infc.phi, &infc.opn, infc.is_phi_eff_owned);
    } else
      infc.phi_eff = phi_eff;
  }

  inline void set_wc(WallBC2D &wc_type, CF_DIM &wc_value) { this->wc_type = &wc_type; this->wc_value = &wc_value; }

  // set BCs
  inline void set_diag_add(double diag_add_scalar)   { diag_add_m_scalar_ = diag_add_scalar;
                                                       diag_add_p_scalar_ = diag_add_scalar; is_matrix_computed_ = false; }

  inline void set_diag_add(double diag_add_m_scalar,
                           double diag_add_p_scalar) { diag_add_m_scalar_ = diag_add_m_scalar;
                                                       diag_add_p_scalar_ = diag_add_p_scalar; is_matrix_computed_ = false; }

  inline void set_diag_add(Vec diag_add)   { diag_add_m_ = diag_add;
                                             diag_add_p_ = diag_add;   is_matrix_computed_ = false; }
  inline void set_diag_add(Vec diag_add_m,
                           Vec diag_add_p) { diag_add_m_ = diag_add_m;
                                             diag_add_p_ = diag_add_p; is_matrix_computed_ = false; }

  inline void set_mu(double mu) { mu_m_ = mu; mu_p_ = mu; variable_mu_ = false; }
  inline void set_mu(double mu_m, double mu_p) { mu_m_ = mu_m; mu_p_ = mu_p;  variable_mu_ = false; }

  void set_mu(Vec mue, DIM( Vec mue_xx = NULL, Vec mue_yy = NULL, Vec mue_zz = NULL ) )
  {
    mue_m_ = mue;
    mue_p_ = mue;

    if (ANDD(mue_xx != NULL, mue_yy != NULL, mue_zz != NULL))
    {
      mue_m_xx_ = mue_xx; mue_m_yy_ = mue_yy; ONLY3D(mue_m_zz_ = mue_zz);
      mue_p_xx_ = mue_xx; mue_p_yy_ = mue_yy; ONLY3D(mue_p_zz_ = mue_zz);
      is_mue_m_dd_owned_ = false;
      is_mue_p_dd_owned_ = false;
    } else {
      compute_mue_dd();
    }

    is_matrix_computed_ = false;

    variable_mu_ = true;
  }

  void set_mu2(Vec mue_m,
               Vec mue_p,
               DIM( Vec mue_m_xx = NULL, Vec mue_m_yy = NULL, Vec mue_m_zz = NULL ),
               DIM( Vec mue_p_xx = NULL, Vec mue_p_yy = NULL, Vec mue_p_zz = NULL ))
  {
    mue_m_ = mue_m;
    mue_p_ = mue_p;

    if (ANDD(mue_m_xx != NULL, mue_m_yy != NULL, mue_m_zz != NULL) &&
        ANDD(mue_p_xx != NULL, mue_p_yy != NULL, mue_p_zz != NULL))
    {
      mue_m_xx_ = mue_m_xx; mue_m_yy_ = mue_m_yy; ONLY3D(mue_m_zz_ = mue_m_zz);
      mue_p_xx_ = mue_p_xx; mue_p_yy_ = mue_p_yy; ONLY3D(mue_p_zz_ = mue_p_zz);
      is_mue_m_dd_owned_ = false;
      is_mue_p_dd_owned_ = false;
    } else {
      compute_mue_dd();
    }

    is_matrix_computed_ = false;

    variable_mu_ = true;
  }

  inline void set_rhs(Vec rhs)   { rhs_m_ = rhs;   rhs_p_ = rhs;   }
  inline void set_rhs(Vec rhs_m,
                      Vec rhs_p) { rhs_m_ = rhs_m; rhs_p_ = rhs_p; }
  inline void set_rhs(CF_DIM &rhs_cf)   { rhs_cf_ = &rhs_cf; }

  inline void set_is_matrix_computed(bool is_matrix_computed) { is_matrix_computed_ = is_matrix_computed; }

  inline void set_tolerances(double rtol, int itmax = PETSC_DEFAULT, double atol = PETSC_DEFAULT, double dtol = PETSC_DEFAULT) {
    ierr = KSPSetTolerances(ksp_, rtol, atol, dtol, itmax); CHKERRXX(ierr);
  }

  inline bool get_matrix_has_nullspace() { return matrix_has_nullspace_; }

  inline void set_first_order_neumann_wall(bool value) { neumann_wall_first_order_  = value; }
  inline void set_use_pointwise_dirichlet (bool value) { use_pointwise_dirichlet_   = value; }
  inline void set_use_taylor_correction   (bool value) { use_taylor_correction_     = value; }
  inline void set_keep_scalling           (bool value) { keep_scalling_             = value; }
  inline void set_kink_treatment          (bool value) { kink_special_treatment_    = value; }
  inline void set_use_sc_scheme           (bool value) { use_sc_scheme_             = value; }
  inline void set_integration_order       (int  value) { integration_order_         = value; }
  inline void set_jump_scheme             (int  value) { jump_scheme_               = value; }
  inline void set_enfornce_diag_scalling  (int  value) { enfornce_diag_scalling_    = value; }
  inline void set_update_ghost_after_solving(bool value) { update_ghost_after_solving_ = value; }

  bool inv_mat2(double *in, double *out);
  bool inv_mat3(double *in, double *out);
  bool inv_mat4(const double m[16], double invOut[16]);

  void find_projection(const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double dxyz_pr[], double &dist_pr, double normal[] = NULL);
  void compute_normal (const double *phi_p, const quad_neighbor_nodes_of_node_t& qnnn, double n[]);
  void get_all_neighbors(const p4est_locidx_t n, p4est_locidx_t *neighbors, bool *neighbor_exists);
  void find_hanging_cells(int *network, bool *hanging_cells);
  void solve(Vec solution, bool use_nonzero_initial_guess = false, KSPType ksp_type = KSPBCGS, PCType pc_type = PCHYPRE);
  void assemble_matrix(Vec solution);

  inline Vec get_mask() { return mask_; }
  inline Vec get_areas() { return areas_; }

  inline void get_phi_dd(std::vector<Vec> *(&phi_dd))
  {
    XCOMP( phi_dd[0] = bdry.phi_xx );
    YCOMP( phi_dd[1] = bdry.phi_yy );
    ZCOMP( phi_dd[2] = bdry.phi_zz );
  }

  inline Vec get_boundary_phi_eff()  { return bdry.phi_eff; }
  inline Vec get_interface_phi_eff() { return infc.phi_eff; }
  inline Mat get_matrix() { return A_; }
  inline std::vector<double>* get_scalling() { return &scalling_; }
  inline void assemble_rhs_only() { setup_linear_system(false, true); }

#ifdef P4_TO_P8
  void reconstruct_domain(std::vector<cube3_mls_t> &cubes);
#else
  void reconstruct_domain(std::vector<cube2_mls_t> &cubes);
#endif


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

#endif // MY_P4EST_POISSON_NODES_MLS_H
